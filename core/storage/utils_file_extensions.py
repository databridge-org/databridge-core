import base64
import magic

def detect_file_type(content: str) -> str:
    """
    Detect file type from content string and return appropriate extension.
    Content can be either base64 encoded or plain text.
    """
    # Try to detect file type from base64 heade
    
    # Decode base64 content
    try:
        decoded_content = base64.b64decode(content)
    except:
        # If not base64, treat as plain text
        decoded_content = content.encode('utf-8')
        
    # Use python-magic to detect mime type from content
    mime = magic.Magic(mime=True)
    detected_type = mime.from_buffer(decoded_content)
    
    # Map mime type to extension
    extension_map = {
        'application/pdf': '.pdf',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'image/tiff': '.tiff',
        'image/bmp': '.bmp',
        'image/svg+xml': '.svg',
        'video/mp4': '.mp4',
        'video/mpeg': '.mpeg',
        'video/quicktime': '.mov',
        'video/x-msvideo': '.avi',
        'video/webm': '.webm',
        'video/x-matroska': '.mkv',
        'video/3gpp': '.3gp',
        'text/plain': '.txt',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
    }
    return extension_map.get(detected_type, '.bin')