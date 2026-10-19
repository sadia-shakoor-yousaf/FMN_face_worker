from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from face_cv.speaker_embed import embed_file
from face_cv.qdrant_gallery import qdrant_settings

vec = embed_file(r'data/voice/wajeeha.ogg')
print('dim', vec.shape)
url, api_key, _ = qdrant_settings('visitor_voices')
client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
client.upsert(
    collection_name='visitor_voices',
    points=[PointStruct(
        id=1,
        vector=vec.tolist(),
        payload={
            'patient_id': 39,
            'name': 'ammi',
            'relationship': 'daughter',
            'embedding_model': 'speechbrain/spkrec-ecapa-voxceleb',
        },
    )],
)
print('upserted')
