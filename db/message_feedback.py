"""Corrections preserve the model label and are scoped to the sender."""
import json
from db.connection import get_connection
from db.feedback_repo import normalize_emotion_label, save_emotion_feedback

def ensure_schema():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE TABLE IF NOT EXISTS message_emotion_feedback (message_id CHAR(32) PRIMARY KEY, correct_label VARCHAR(20) NOT NULL, updated_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4')
        conn.commit()

def correct_message(room_id, message_id, user_id, username, label):
    label=normalize_emotion_label(label)
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT sender_id,text,emotion FROM chat_messages WHERE id=%s AND room_id=%s',(message_id,room_id))
            row=cur.fetchone()
            if not row or row[0]!=user_id:
                raise PermissionError('只能更正自己發出的訊息')
            if not row[2]: raise ValueError('請等候情緒分析完成')
            cur.execute('SELECT correct_label FROM message_emotion_feedback WHERE message_id=%s',(message_id,))
            old=cur.fetchone()
            if old and old[0]==label: return label
            if label==row[2]: raise ValueError('請選擇與原判斷不同的情緒')
            # New correction events supersede older ones in correction memory.
            save_emotion_feedback(username,json.dumps({'text':row[1],'history':[],'message_id':message_id},ensure_ascii=False),row[2],label)
            cur.execute('INSERT INTO message_emotion_feedback (message_id,correct_label) VALUES (%s,%s) ON DUPLICATE KEY UPDATE correct_label=VALUES(correct_label)',(message_id,label))
        conn.commit()
    return label

def corrections_for(room_id):
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT f.message_id,f.correct_label FROM message_emotion_feedback f JOIN chat_messages m ON m.id=f.message_id WHERE m.room_id=%s',(room_id,))
            return dict(cur.fetchall())
