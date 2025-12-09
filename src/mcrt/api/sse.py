"""Server-Sent Events (SSE) support for streaming simulation updates."""

import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator

from sse_starlette.sse import EventSourceResponse

from mcrt.simulation.progressive import BatchUpdate


async def format_sse_event(event_type: str, data: dict) -> str:
    """Format data as SSE event.

    Args:
        event_type: Event type name
        data: Event data dict

    Returns:
        Formatted SSE event string
    """
    json_data = json.dumps(data)
    return f"event: {event_type}\ndata: {json_data}\n\n"


class SSEManager:
    """Manages SSE connections and event broadcasting."""

    def __init__(self):
        self._connections: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Subscribe to events for a session.

        Args:
            session_id: Session to subscribe to

        Returns:
            Queue that will receive events
        """
        if session_id not in self._connections:
            self._connections[session_id] = []

        queue: asyncio.Queue = asyncio.Queue()
        self._connections[session_id].append(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue):
        """Unsubscribe from session events.

        Args:
            session_id: Session to unsubscribe from
            queue: Queue to remove
        """
        if session_id in self._connections:
            try:
                self._connections[session_id].remove(queue)
            except ValueError:
                pass

            if not self._connections[session_id]:
                del self._connections[session_id]

    async def broadcast(self, session_id: str, event_type: str, data: dict):
        """Broadcast event to all subscribers of a session.

        Args:
            session_id: Session to broadcast to
            event_type: Event type name
            data: Event data
        """
        if session_id not in self._connections:
            return

        for queue in self._connections[session_id]:
            await queue.put({"event": event_type, "data": data})

    async def send_batch_update(self, session_id: str, update: BatchUpdate):
        """Send batch update event.

        Args:
            session_id: Session ID
            update: BatchUpdate object
        """
        await self.broadcast(
            session_id,
            "batch_complete",
            {
                "batch_number": update.batch_number,
                "total_batches": update.total_batches,
                "photons_completed": update.photons_completed,
                "photons_target": update.photons_target,
                "progress_percent": (
                    100 * update.photons_completed / update.photons_target
                    if update.photons_target > 0
                    else 0
                ),
                "results": {
                    "wavelength_um": update.wavelength_um,
                    "reflectance": update.reflectance,
                    "absorptance": update.absorptance,
                    "transmittance": update.transmittance,
                },
                "timestamp": update.timestamp,
            },
        )

    async def send_completion(
        self, session_id: str, status: str, results: dict | None = None
    ):
        """Send simulation completion event.

        Args:
            session_id: Session ID
            status: Final status
            results: Final results
        """
        await self.broadcast(
            session_id,
            "simulation_complete",
            {
                "session_id": session_id,
                "status": status,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def send_error(self, session_id: str, error: str):
        """Send error event.

        Args:
            session_id: Session ID
            error: Error message
        """
        await self.broadcast(
            session_id,
            "error",
            {
                "session_id": session_id,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            },
        )


async def event_generator(
    queue: asyncio.Queue, session_id: str, sse_manager: SSEManager
) -> AsyncGenerator[dict, None]:
    """Generate SSE events from queue.

    Args:
        queue: Event queue
        session_id: Session ID for cleanup
        sse_manager: SSE manager for cleanup

    Yields:
        Event dicts for SSE response
    """
    try:
        while True:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(queue.get(), timeout=30.0)

                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"]),
                }

                # Check for completion events
                if event["event"] in ("simulation_complete", "error"):
                    break

            except asyncio.TimeoutError:
                # Send keepalive
                yield {"event": "keepalive", "data": ""}

    finally:
        sse_manager.unsubscribe(session_id, queue)


def create_sse_response(
    queue: asyncio.Queue, session_id: str, sse_manager: SSEManager
) -> EventSourceResponse:
    """Create SSE response for a session.

    Args:
        queue: Event queue
        session_id: Session ID
        sse_manager: SSE manager

    Returns:
        EventSourceResponse for streaming
    """
    return EventSourceResponse(
        event_generator(queue, session_id, sse_manager),
        media_type="text/event-stream",
    )
