from langchain_redis import RedisChatMessageHistory

def get_history(redis_url, session_id):
    return RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
