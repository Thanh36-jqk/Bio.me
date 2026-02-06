"""
WebSocket Manager for Real-time Biometric Processing Updates
Provides live progress updates during enrollment and verification
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    WebSocket connection manager for real-time updates
    
    Features:
    - Multi-user connection handling
    - Progress updates during biometric processing
    - Task status synchronization
    - Automatic reconnection handling
    """
    
    def __init__(self):
        # Store active connections per user
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Accept new WebSocket connection
        
        Args:
            websocket: WebSocket connection
            user_id: User identifier
        """
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        logger.info(f"✓ WebSocket connected for user: {user_id}")
        
        # Send initial connection confirmation
        await self.send_personal_message(
            json.dumps({
                'type': 'connection',
                'status': 'connected',
                'message': 'Real-time updates active'
            }),
            websocket
        )
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """
        Disconnect WebSocket
        
        Args:
            websocket: WebSocket to disconnect
            user_id: User identifier
        """
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            
            # Remove empty connection list
            if len(self.active_connections[user_id]) == 0:
                del self.active_connections[user_id]
        
        logger.info(f"✗ WebSocket disconnected for user: {user_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """
        Send message to specific WebSocket
        
        Args:
            message: JSON string message
            websocket: Target WebSocket
        """
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def send_to_user(self, message: dict, user_id: str):
        """
        Send message to all connections of a user
        
        Args:
            message: Dictionary to send (will be JSON encoded)
            user_id: Target user
        """
        if user_id in self.active_connections:
            message_str = json.dumps(message)
            dead_connections = []
            
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    logger.error(f"Failed to send to {user_id}: {e}")
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for conn in dead_connections:
                self.disconnect(conn, user_id)
    
    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected users
        
        Args:
            message: Dictionary to broadcast
        """
        message_str = json.dumps(message)
        
        for user_id, connections in list(self.active_connections.items()):
            for connection in connections:
                try:
                    await connection.send_text(message_str)
                except Exception:
                    pass
    
    # ========== Biometric-specific Messages ==========
    
    async def send_processing_update(self, 
                                    user_id: str, 
                                    step: str, 
                                    progress: float,
                                    message: str = ""):
        """
        Send biometric processing progress update
        
        Args:
            user_id: User identifier
            step: Current processing step
            progress: Progress percentage (0.0 - 1.0)
            message: Optional status message
        """
        await self.send_to_user({
            'type': 'processing_update',
            'step': step,
            'progress': progress,
            'message': message
        }, user_id)
    
    async def send_recognition_result(self,
                                     user_id: str,
                                     biometric_type: str,
                                     success: bool,
                                     confidence: float = None,
                                     matched_user: str = None):
        """
        Send recognition result
        
        Args:
            user_id: User identifier
            biometric_type: 'face', 'iris', or 'fingerprint'
            success: Recognition success
            confidence: Match confidence (0.0 - 1.0)
            matched_user: Matched username if successful
        """
        await self.send_to_user({
            'type': 'recognition_result',
            'biometric_type': biometric_type,
            'success': success,
            'confidence': confidence,
            'matched_user': matched_user
        }, user_id)
    
    async def send_enrollment_status(self,
                                    user_id: str,
                                    biometric_type: str,
                                    status: str,
                                    num_samples: int = 0):
        """
        Send enrollment status update
        
        Args:
            user_id: User identifier
            biometric_type: Type of biometric
            status: 'in_progress', 'completed', 'failed'
            num_samples: Number of samples processed
        """
        await self.send_to_user({
            'type': 'enrollment_status',
            'biometric_type': biometric_type,
            'status': status,
            'num_samples': num_samples
        }, user_id)
    
    async def send_error(self, user_id: str, error_code: str, error_message: str):
        """
        Send error message
        
        Args:
            user_id: User identifier
            error_code: Error code
            error_message: Human-readable error message
        """
        await self.send_to_user({
            'type': 'error',
            'error_code': error_code,
            'message': error_message
        }, user_id)
    
    def get_active_users(self) -> List[str]:
        """Get list of users with active connections"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(conns) for conns in self.active_connections.values())

# Global connection manager instance
ws_manager = ConnectionManager()
