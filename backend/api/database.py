"""
MongoDB Database Manager for User Authentication
Enterprise-grade biometric user management
"""
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, List
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self, connection_string: str = None):
        """
        Initialize MongoDB manager
        
        Args:
            connection_string: MongoDB connection URI
                               Default: mongodb://localhost:27017
        """
        self.connection_string = connection_string or os.getenv(
            "MONGODB_URI", 
            "mongodb://localhost:27017"
        )
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.users_collection = None
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            # Test connection
            await self.client.admin.command('ping')
            
            #Select database
            self.db = self.client['biometric_mfa']
            self.users_collection = self.db['users']
            
            # Create indexes
            await self.users_collection.create_index("username", unique=True)
            
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def create_user(self, username: str) -> str:
        """
        Create a new user
        
        Args:
            username: Unique username
        
        Returns:
            str: User ID
        
        Raises:
            Exception: If username already exists
        """
        # Check if user exists
        existing = await self.users_collection.find_one({"username": username})
        if existing:
            raise Exception(f"Username '{username}' already exists")
        
        user_doc = {
            "username": username,
            "face_registered": False,
            "iris_registered": False,
            "fingerprint_registered": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await self.users_collection.insert_one(user_doc)
        logger.info(f"Created user: {username}")
        return str(result.inserted_id)
    
    async def get_user(self, username: str) -> Optional[Dict]:
        """
        Get user by username
        
        Args:
            username: Username to lookup
        
        Returns:
            Dict with user info or None
        """
        user = await self.users_collection.find_one({"username": username})
        if user:
            user['_id'] = str(user['_id'])
            return user
        return None
    
    async def update_user_biometric(self, username: str, biometric_type: str, status: bool):
        """
        Update biometric registration status
        
        Args:
            username: Username
            biometric_type: 'face', 'iris', or 'fingerprint'
            status: True if registered, False otherwise
        """
        valid_types = ['face', 'iris', 'fingerprint']
        if biometric_type not in valid_types:
            raise ValueError(f"Invalid biometric type: {biometric_type}")
        
        field = f"{biometric_type}_registered"
        
        result = await self.users_collection.update_one(
            {"username": username},
            {
                "$set": {
                    field: status,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"Updated {biometric_type} status for {username}: {status}")
        return result.modified_count > 0
    
    async def delete_user(self, username: str) -> bool:
        """
        Delete user
        
        Args:
            username: Username to delete
        
        Returns:
            bool: True if deleted, False if not found
        """
        result = await self.users_collection.delete_one({"username": username})
        if result.deleted_count > 0:
            logger.info(f"Deleted user: {username}")
        return result.deleted_count > 0
    
    async def list_users(self) -> List[Dict]:
        """List all users"""
        cursor = self.users_collection.find().sort("created_at", -1)
        users = await cursor.to_list(length=100)
        
        for user in users:
            user['_id'] = str(user['_id'])
        
        return users
    
    async def get_user_count(self) -> int:
        """Get total number of users"""
        return await self.users_collection.count_documents({})
    
    async def get_stats(self) -> Dict:
        """Get system statistics"""
        total_users = await self.get_user_count()
        
        face_count = await self.users_collection.count_documents({"face_registered": True})
        iris_count = await self.users_collection.count_documents({"iris_registered": True})
        fp_count = await self.users_collection.count_documents({"fingerprint_registered": True})
        
        return {
            "total_users": total_users,
            "face_enrolled": face_count,
            "iris_enrolled": iris_count,
            "fingerprint_enrolled": fp_count
        }

# Global instance
db_manager = MongoDBManager()
