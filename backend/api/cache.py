"""
Redis Cache Manager for Biometric System
High-performance caching layer for embeddings and session data
"""
import redis.asyncio as aioredis
import pickle
import json
import logging
from typing import Optional, Any, Dict
import numpy as np
from datetime import timedelta

logger = logging.getLogger(__name__)

class RedisCacheManager:
    """
    Redis caching layer for biometric embeddings and user sessions
    
    Features:
    - Async operations
    - Automatic TTL management
    - Embedding serialization (NumPy arrays)
    - Session storage
    - Performance metrics caching
    """
    
    def __init__(self, redis_url="redis://localhost:6379"):
        """
        Initialize Redis cache manager
        
        Args:
            redis_url: Redis connection string
        """
        self.redis_url = redis_url
        self.client: Optional[aioredis.Redis] = None
        self.default_ttl = 3600  # 1 hour
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False  # We handle binary data
            )
            # Test connection
            await self.client.ping()
            logger.info(f"✓ Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis")
    
    # ========== Embedding Cache ==========
    
    async def cache_embedding(self, 
                             user_id: str, 
                             biometric_type: str,
                             embedding: np.ndarray,
                             ttl: int = None):
        """
        Cache biometric embedding
        
        Args:
            user_id: User identifier
            biometric_type: 'face', 'iris', or 'fingerprint'
            embedding: NumPy array embedding
            ttl: Time to live in seconds (default: 1 hour)
        """
        try:
            key = f"emb:{biometric_type}:{user_id}"
            value = pickle.dumps(embedding)
            ttl = ttl or self.default_ttl
            
            await self.client.setex(key, ttl, value)
            logger.debug(f"Cached {biometric_type} embedding for {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
    
    async def get_embedding(self, 
                           user_id: str, 
                           biometric_type: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding
        
        Args:
            user_id: User identifier
            biometric_type: 'face', 'iris', or 'fingerprint'
        
        Returns:
            NumPy array or None if not found
        """
        try:
            key = f"emb:{biometric_type}:{user_id}"
            data = await self.client.get(key)
            
            if data:
                embedding = pickle.loads(data)
                logger.debug(f"Retrieved {biometric_type} embedding for {user_id} from cache")
                return embedding
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    # ========== Session Management ==========
    
    async def create_session(self, 
                            user_id: str, 
                            session_data: Dict,
                            ttl: int = 86400):  # 24 hours
        """
        Create user session
        
        Args:
            user_id: User identifier
            session_data: Session information
            ttl: Session lifetime (default: 24 hours)
        """
        try:
            key = f"session:{user_id}"
            value = json.dumps(session_data)
            
            await self.client.setex(key, ttl, value)
            logger.info(f"Created session for {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
    
    async def get_session(self, user_id: str) -> Optional[Dict]:
        """Get user session"""
        try:
            key = f"session:{user_id}"
            data = await self.client.get(key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def delete_session(self, user_id: str):
        """Delete user session"""
        try:
            key = f"session:{user_id}"
            await self.client.delete(key)
            logger.info(f"Deleted session for {user_id}")
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
    
    # ========== Rate Limiting ==========
    
    async def check_rate_limit(self, 
                              identifier: str, 
                              max_requests: int = 100,
                              window: int = 60) -> bool:
        """
        Check if request is within rate limit
        
        Args:
            identifier: User ID or IP address
            max_requests: Maximum requests per window
            window: Time window in seconds
        
        Returns:
            True if allowed, False if rate limited
        """
        try:
            key = f"rate:{identifier}"
            
            # Increment counter
            count = await self.client.incr(key)
            
            # Set expiry on first request
            if count == 1:
                await self.client.expire(key, window)
            
            return count <= max_requests
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Fail open
    
    # ========== Performance Metrics ==========
    
    async def cache_metric(self, 
                          metric_name: str, 
                          value: Any,
                          ttl: int = 300):  # 5 minutes
        """Cache performance metric"""
        try:
            key = f"metric:{metric_name}"
            await self.client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Failed to cache metric: {e}")
    
    async def get_metric(self, metric_name: str) -> Optional[Any]:
        """Get cached metric"""
        try:
            key = f"metric:{metric_name}"
            data = await self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Failed to get metric: {e}")
            return None
    
    # ========== Utility Methods ==========
    
    async def flush_all(self):
        """Flush all cached data (use with caution!)"""
        try:
            await self.client.flushall()
            logger.warning("Flushed all Redis data")
        except Exception as e:
            logger.error(f"Failed to flush Redis: {e}")
    
    async def get_stats(self) -> Dict:
        """Get Redis statistics"""
        try:
            info = await self.client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0'),
                'total_keys': await self.client.dbsize()
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

# Global instance
cache_manager = RedisCacheManager()
