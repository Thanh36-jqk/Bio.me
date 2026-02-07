"""
MongoDB Index Cleanup Script
Run this ONCE to remove the old username index that causes E11000 errors
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def fix_database():
    """Drop old username index and create new email index"""
    
    # Get connection string
    connection_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    
    print(f"Connecting to MongoDB: {connection_string}")
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(connection_string)
    db = client['biometric_mfa']
    users_collection = db['users']
    
    try:
        # List all indexes
        print("\n[INFO] Current indexes:")
        indexes = await users_collection.index_information()
        for index_name, index_info in indexes.items():
            print(f"  - {index_name}: {index_info}")
        
        # Drop the old username_1 index
        if "username_1" in indexes:
            print("\n[ACTION] Dropping 'username_1' index...")
            await users_collection.drop_index("username_1")
            print("[OK] Successfully dropped 'username_1' index!")
        else:
            print("\n[INFO] 'username_1' index not found (already removed or never existed)")
        
        # Create email index if it doesn't exist
        if "email_1" not in indexes:
            print("\n[ACTION] Creating 'email_1' unique index...")
            await users_collection.create_index("email", unique=True)
            print("[OK] Successfully created 'email_1' index!")
        else:
            print("\n[INFO] 'email_1' index already exists")
        
        # List final indexes
        print("\n[INFO] Final indexes:")
        final_indexes = await users_collection.index_information()
        for index_name, index_info in final_indexes.items():
            print(f"  - {index_name}: {index_info}")
        
        print("\n" + "="*60)
        print("[SUCCESS] Database cleanup completed!")
        print("You can now register users without E11000 errors.")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to fix database: {e}")
        print("\nIf you see authentication errors, make sure MONGODB_URI is set correctly.")
        print("For local MongoDB: mongodb://localhost:27017")
    
    finally:
        client.close()

if __name__ == "__main__":
    print("="*60)
    print("MongoDB Index Cleanup Tool")
    print("="*60)
    asyncio.run(fix_database())
