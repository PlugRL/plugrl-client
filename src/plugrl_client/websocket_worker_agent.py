import time
import enum
from typing import Dict, Optional, Tuple

from typing_extensions import override
from loguru import logger
import websockets.sync.client
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from plugrl_client import base_agent as _base_agent
from plugrl_client import msgpack_numpy

SERVER_STOP_REASON = "plugrl-server-stop"


def _get_close_details(exc: ConnectionClosed) -> tuple[int | None, str]:
    if exc.rcvd is not None:
        return exc.rcvd.code, exc.rcvd.reason
    if exc.sent is not None:
        return exc.sent.code, exc.sent.reason
    return None, ""


class ServerStopped(RuntimeError):
    """Raised when the server explicitly requests workers to stop."""

    pass

class MessageType(enum.Enum):
    INFER = "infer"
    FEEDBACK = "feedback"
    METADATA = "metadata"
    ACTION = "action"
    def __str__(self):
        return self.value

class WebSocketWorkerAgent(_base_agent.BaseAgent):
    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        # Initial connection upon agent creation
        self._ws, self._server_metadata = self._wait_for_server()

    def _close_connection(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        except Exception:
            pass
        finally:
            self._ws = None
        
    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """Blocks until a connection to the server is established and metadata is received."""
        RECONNECT_DELAY = 5  # seconds
        logger.info(f"Waiting for server at {self._uri}...")
        
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                logger.info("Connection established. Awaiting server metadata.")
                
                # First message must be METADATA
                metadata_msg = msgpack_numpy.unpackb(conn.recv())
                if metadata_msg.get("message_type") != str(MessageType.METADATA):
                     logger.warning(f"Expected METADATA but received {metadata_msg.get('message_type')}. Reconnecting.")
                     conn.close()
                     raise ConnectionRefusedError # Force retry
                     
                logger.info("Successfully received server metadata.")
                return conn, metadata_msg["data"]
                
            except ConnectionClosedOK as e:
                close_code, close_reason = _get_close_details(e)
                if close_reason == SERVER_STOP_REASON:
                    raise ServerStopped(
                        "Server requested worker shutdown while reconnecting."
                    ) from e
                logger.warning(
                    "Server closed the connection during metadata exchange. "
                    f"Retrying in {RECONNECT_DELAY} seconds... code={close_code}, "
                    f"reason={close_reason or '<empty>'}"
                )
                time.sleep(RECONNECT_DELAY)
            except (ConnectionRefusedError, TimeoutError, ConnectionClosedError):
                logger.warning(f"Server not available or connection failed. Retrying in {RECONNECT_DELAY} seconds...")
                time.sleep(RECONNECT_DELAY)
            except Exception as e:
                # Catch other errors during initialization (e.g., malformed metadata)
                logger.error(f"Error during initial connection or metadata exchange: {e}")
                time.sleep(RECONNECT_DELAY)
    
    def _ensure_connection(self) -> None:
        """Checks connection status and blocks for reconnection if closed."""
        if not self._ws:
            logger.warning("Connection closed. Attempting to re-establish connection.")
            # Re-run the full connection and metadata exchange
            self._ws, self._server_metadata = self._wait_for_server()

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """Sends an observation and waits for the action, retrying on connection failure."""
        while True:
            self._ensure_connection()
            ws = self._ws
            if ws is None:
                raise RuntimeError("WebSocket connection is not available.")
            try:
                # 1. Send INFER
                packed_data = self._packer.pack(dict(message_type=str(MessageType.INFER), data=obs))
                ws.send(packed_data)
                
                # 2. Receive ACTION
                response = ws.recv()
                
                if isinstance(response, str):
                    # The server sent a string, indicating a server error.
                    raise RuntimeError(f"Error in inference server:\n{response}")
                
                # Success
                return msgpack_numpy.unpackb(response)["data"]
            except ConnectionClosedOK as e:
                close_code, close_reason = _get_close_details(e)
                self._close_connection()
                if close_reason == SERVER_STOP_REASON:
                    raise ServerStopped(
                        "Server requested worker shutdown after algorithm stop."
                    ) from e
                logger.warning(
                    "Connection closed normally during INFER/ACTION exchange. "
                    f"Waiting for server to come back and retrying... code={close_code}, "
                    f"reason={close_reason or '<empty>'}"
                )
                continue
                
            except ConnectionClosedError as e:
                logger.warning(f"Connection closed during INFER/ACTION exchange. Error: {e}")
                self._close_connection()
                continue # Retry the entire operation
            except Exception:
                self._close_connection()
                raise

    @override
    def feedback(self, obs: Dict, rewards: float, terminated: bool, truncated: bool, info: Dict) -> None:
        """Sends feedback data, retrying on connection failure."""
        while True:
            self._ensure_connection()
            ws = self._ws
            if ws is None:
                raise RuntimeError("WebSocket connection is not available.")
            try:
                packed_data = self._packer.pack(dict(
                    message_type=str(MessageType.FEEDBACK),
                    data=dict(obs=obs, rewards=rewards, terminated=terminated, truncated=truncated, info=info)
                ))
                ws.send(packed_data)
                
                # Success
                return
            except ConnectionClosedOK as e:
                close_code, close_reason = _get_close_details(e)
                self._close_connection()
                if close_reason == SERVER_STOP_REASON:
                    raise ServerStopped(
                        "Server requested worker shutdown after algorithm stop."
                    ) from e
                logger.warning(
                    "Connection closed normally during FEEDBACK send. "
                    f"Waiting for server to come back and retrying... code={close_code}, "
                    f"reason={close_reason or '<empty>'}"
                )
                continue
            except ConnectionClosedError as e:
                logger.warning(f"Connection closed during FEEDBACK send. Error: {e}")
                self._close_connection()
                continue # Retry the send operation
            except Exception:
                self._close_connection()
                raise

    @override
    def reset(self) -> None:
        pass