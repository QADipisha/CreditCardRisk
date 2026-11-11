"""SQLite-backed conversation state with bounded history and no raw sensitive profile storage."""
import json, sqlite3, os, time
from pathlib import Path
class StateStore:
    def __init__(self,path=None):
        self.path=path or os.getenv('STATE_DB','data/conversations.sqlite3'); Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as c: c.execute('CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, state TEXT NOT NULL, updated REAL NOT NULL)')
    def get(self,session_id):
        with sqlite3.connect(self.path) as c: row=c.execute('SELECT state FROM sessions WHERE session_id=?',(session_id,)).fetchone()
        return json.loads(row[0]) if row else {"turns":[],"last_workflow":None}
    def save(self,session_id,state):
        state['turns']=state.get('turns',[])[-12:]
        with sqlite3.connect(self.path) as c: c.execute('INSERT INTO sessions VALUES(?,?,?) ON CONFLICT(session_id) DO UPDATE SET state=excluded.state, updated=excluded.updated',(session_id,json.dumps(state,ensure_ascii=False),time.time()))
