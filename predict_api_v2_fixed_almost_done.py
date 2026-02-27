"""
Sign-Bridge AI — Cloud-Ready FastAPI Backend
=============================================
Architecture (browser-camera edition):
  • WS  /ws/camera  — browser streams frames → server runs MediaPipe+TF → returns annotated JPEG
  • WS  /ws         — server PUSHES state the moment it changes
  • POST /command   — UI buttons → engine actions
  • POST /speak_translation — gTTS audio returned as base64 MP3 → browser plays it
  • GET  /          — serves the complete HTML UI

Run:
    uvicorn predict_api_v2:app --host 0.0.0.0 --port 8000

    if the 0.0.0.0:8000 is not working than try out the localhost:8000 this will works because the DNS issue

    also here i have provided the .yml -> that is the conda requirements so if you have to use you have to make the conda environment and do the
    install dependencies it has
"""

import asyncio, threading, time, queue, os, pickle, json, io, base64
from contextlib import asynccontextmanager
import cv2, numpy as np
import mediapipe as mp
import tensorflow as tf
import language_tool_python
from collections import deque, Counter
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════
app = FastAPI(title="Sign-Bridge AI")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
MODEL_PATH   = "asl_landmark_model.keras"
ENCODER_PATH = "label_encoder.json"
WS_CACHE     = "word_suggester_cache.pkl"

CONF_THRESH  = 0.75
SMOOTH_WIN   = 10
HOLD_DUR     = 1.2
LETTER_HOLD  = 1.0
LETTER_COOL  = 1.5

CLS_SPACE = "SPACE"
CLS_DOT   = "DOT"
CLS_AC    = "AUTOCORRECT"
SPECIALS  = {CLS_SPACE, CLS_DOT, CLS_AC}

FD = cv2.FONT_HERSHEY_DUPLEX
FS = cv2.FONT_HERSHEY_SIMPLEX
TOP_H = 140

DARK=(22,22,22); SEP=(55,55,55); WHITE=(240,240,240)
CYAN=(80,210,255); YELLOW=(210,195,65); GREEN=(75,220,75)
BLUE=(75,185,255); ORANGE=(255,165,55); PINK=(255,115,175)
PURPLE=(175,135,255); TEAL=(60,200,180); BLACK_C=(10,10,10)
CHIP_BG=(28,65,28); CHIP_TXT=(110,225,110); CHIP_BOR=(50,95,50)

_lock        = threading.Lock()
_model_ready = False
_tools_ready = False

# ════════════════════════════════════════════════════════
# WEBSOCKET MANAGER
# ════════════════════════════════════════════════════════
class WSManager:
    def __init__(self):
        self._clients: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:    await ws.send_text(msg)
                except: dead.append(ws)
            for ws in dead: self._clients.remove(ws)

ws_manager = WSManager()

# ════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════
print("[INFO] Loading ASL model …")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH) as f:
        class_names = json.load(f)
    _model_ready = True
    print(f"[INFO] Model ready — {len(class_names)} classes")
except Exception as e:
    model = None; class_names = []
    print(f"[ERROR] Model: {e}")

# ════════════════════════════════════════════════════════
# TTS — gTTS → base64 MP3 → sent to browser via WebSocket
# No PowerShell, no OS dependency, works on any cloud server
# ════════════════════════════════════════════════════════
_tts_audio_queue: queue.Queue = queue.Queue(maxsize=5)

def _text_to_base64_mp3(text: str, lang: str = "en") -> Optional[str]:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[TTS] gTTS error: {e}")
        return None

def _queue_speak(text: str, lang: str = "en"):
    def _gen():
        audio_b64 = _text_to_base64_mp3(text, lang=lang)
        if audio_b64:
            try: _tts_audio_queue.put_nowait({"type": "tts_audio", "data": audio_b64})
            except queue.Full: pass
    threading.Thread(target=_gen, daemon=True).start()

# ════════════════════════════════════════════════════════
# SPELL + LANGUAGETOOL
# ════════════════════════════════════════════════════════
spell = lang_tool = None
_tools_event = threading.Event()

def _load_tools():
    global spell, lang_tool, _tools_ready
    res = {}
    def _ls():
        try:
            from spellchecker import SpellChecker
            res['s'] = SpellChecker(); print("[INFO] SpellChecker ✓")
        except ImportError: res['s'] = None
    def _ll():
        try:
            res['l'] = language_tool_python.LanguageTool('en-US', remote_server=None)
            print("[INFO] LanguageTool ✓")
        except Exception as e:
            res['l'] = None; print(f"[WARN] LangTool: {e}")
    t1=threading.Thread(target=_ls,daemon=True)
    t2=threading.Thread(target=_ll,daemon=True)
    t1.start(); t2.start(); t1.join(); t2.join()
    spell=res.get('s'); lang_tool=res.get('l')
    _tools_event.set(); _tools_ready=True
    print("[INFO] Tools ready ✓")

threading.Thread(target=_load_tools, daemon=True).start()

# ════════════════════════════════════════════════════════
# MEDIAPIPE
# ════════════════════════════════════════════════════════
mph = mp.solutions.hands
mpd = mp.solutions.drawing_utils
mps = mp.solutions.drawing_styles
detector = mph.Hands(static_image_mode=False, max_num_hands=1,
                     model_complexity=1, min_detection_confidence=0.6,
                     min_tracking_confidence=0.5)

def extract_landmarks(res):
    if not res.multi_hand_landmarks: return None
    lms = res.multi_hand_landmarks[0].landmark
    c   = np.array([[l.x,l.y,l.z] for l in lms], dtype=np.float32)
    c  -= c[0]; c /= (np.max(np.abs(c))+1e-6)
    return c.flatten()

# ════════════════════════════════════════════════════════
# SMOOTHER / HOLD / LETTER TRACKERS
# ════════════════════════════════════════════════════════
class Smoother:
    def __init__(self,w): self.buf=deque(maxlen=w)
    def push(self,p):     self.buf.append(p)
    def reset(self):      self.buf.clear()
    def top(self):
        if not self.buf: return 0,0.0
        a=np.mean(self.buf,axis=0); i=np.argmax(a)
        return i,a[i]

class HoldTracker:
    def __init__(self,dur=1.2):
        self.dur=dur; self.lbl=self.t0=None; self.fired=False
    def update(self,lbl):
        now=time.time()
        if lbl is None:
            self.lbl=self.t0=None; self.fired=False; return False
        if lbl!=self.lbl:
            self.lbl=lbl; self.t0=now; self.fired=False; return False
        if not self.fired and self.t0 and (now-self.t0)>=self.dur:
            self.fired=True; return True
        return False
    def prog(self):
        if not self.lbl or not self.t0: return 0.0
        return min((time.time()-self.t0)/self.dur,1.0)

class LetterTracker:
    def __init__(self,hd=1.0,cd=1.5):
        self.hd=hd; self.cd=cd; self.lbl=self.t0=None
        self.last={}; self.fired=False
    def update(self,lbl):
        now=time.time()
        if lbl!=self.lbl:
            self.lbl=lbl; self.t0=now; self.fired=False; return None
        if self.fired: return None
        if not self.t0 or (now-self.t0)<self.hd: return None
        if (now-self.last.get(lbl,0))<self.cd: return None
        self.fired=True; self.last[lbl]=now; return lbl
    def prog(self):
        if not self.lbl or not self.t0: return 0.0
        return min((time.time()-self.t0)/self.hd,1.0)
    def reset(self):
        self.lbl=self.t0=None; self.fired=False

smoother = Smoother(SMOOTH_WIN)
hold      = HoldTracker(HOLD_DUR)
lt        = LetterTracker(LETTER_HOLD, LETTER_COOL)

# ════════════════════════════════════════════════════════
# WORD SUGGESTER
# ════════════════════════════════════════════════════════
class WordSuggester:
    def __init__(self):
        if os.path.exists(WS_CACHE):
            try:
                d=pickle.load(open(WS_CACHE,'rb'))
                self.freq=d['freq']; self.bg=d['bg']; self.vocab=d['vocab']
                print(f"[INFO] WordSuggester ✓ ({len(self.vocab):,} words, cached)")
                return
            except: pass
        try:
            from nltk.corpus import brown
            ws=[w.lower() for w in brown.words() if w.isalpha()]
            self.freq=Counter(ws); self.bg=Counter(zip(ws[:-1],ws[1:]))
            self.vocab=sorted(self.freq)
            pickle.dump({'freq':self.freq,'bg':self.bg,'vocab':self.vocab},open(WS_CACHE,'wb'))
            print(f"[INFO] WordSuggester ✓ ({len(self.vocab):,} words)")
        except Exception as e:
            self.freq=Counter(); self.bg=Counter(); self.vocab=[]
            print(f"[WARN] WordSuggester: {e}")

    def suggest(self,cw,pw='',n=5):
        cw=cw.lower(); pw=pw.lower()
        if not self.vocab: return []
        if cw:
            m=[w for w in self.vocab if w.startswith(cw) and w!=cw]
            m.sort(key=lambda w:self.freq.get(w,0),reverse=True)
            return m[:n]
        if pw:
            c=[(w2,k) for (w1,w2),k in self.bg.items() if w1==pw]
            c.sort(key=lambda x:x[1],reverse=True)
            return [w for w,_ in c[:n]]
        return [w for w,_ in self.freq.most_common(n)]

_ws_model = WordSuggester()
word_suggs: List[str] = []

def update_suggs():
    global word_suggs
    word_suggs=_ws_model.suggest(sb.cw,sb.words[-1] if sb.words else '',n=5)

def apply_sugg(i:int):
    if not word_suggs or i>=len(word_suggs): return
    sb.cw=word_suggs[i]; sb._s(f"Word: '{sb.cw}'"); update_suggs()

# ════════════════════════════════════════════════════════
# SENTENCE BUILDER
# ════════════════════════════════════════════════════════
class SentenceBuilder:
    def __init__(self):
        self.cw=''; self.words=[]; self.sents=[]
        self.suggs=[]; self.show_ac=False
        self.status='Ready — start signing!'; self.st=time.time()

    def _s(self,m): self.status=m; self.st=time.time(); _notify_state()

    def add(self,ch):
        self.cw+=ch; self._s(f'Typing: {self.cw}'); update_suggs()

    def space(self):
        if self.cw.strip():
            self.words.append(self.cw); self._s(f"Saved: '{self.cw}'")
            self.cw=''; update_suggs()
        else: self._s('Nothing to space')

    def dot(self):
        if self.cw.strip():
            self.words.append(self.cw); self.cw=''
        if self.words:
            fin=(' '.join(self.words))+'.'; self.sents.append(fin)
            self.words=[]; self._s('● Sentence done — speaking …')
            _queue_speak(fin); update_suggs()
        else: self._s('Nothing to end')

    def backspace(self):
        if self.cw:
            ch=self.cw[-1]; self.cw=self.cw[:-1]
            self._s(f"Del '{ch}'"); update_suggs()
        elif self.words:
            self.cw=self.words.pop(); self._s(f"Restored: '{self.cw}'"); update_suggs()
        else: self._s('Nothing to delete')

    def clear(self):
        self.__init__(); self._s('Cleared ✓')

    def pick(self,i:int):
        if self.suggs and 0<=i<len(self.suggs):
            ch=self.suggs[i]
            if self.sents: self.sents[-1]=ch
            else:          self.sents.append(ch)
            self.suggs=[]; self.show_ac=False
            self._s(f'✓ [{i+1}] applied — speaking corrected …')
            _queue_speak(ch)

    def live(self):
        return ' '.join(self.words+([self.cw] if self.cw else []))

    def history(self,n=3):
        return self.sents[-n:]

sb = SentenceBuilder()

# ════════════════════════════════════════════════════════
# AUTOCORRECT
# ════════════════════════════════════════════════════════
ac_busy = False

def _sw(word):
    if not spell: return word
    w=word.lower().strip('.,!?')
    if len(w)<=1: return word
    f=spell.correction(w)
    if not f: return word
    return f.capitalize() if word[0].isupper() else f

def _clean(t):
    t=t.strip()
    while '  ' in t: t=t.replace('  ',' ')
    return t[0].upper()+t[1:] if t else t

def _correct(sentence):
    import re as _re
    sp=_clean(' '.join(_sw(w) for w in sentence.split()))
    if lang_tool:
        try:
            gr=language_tool_python.utils.correct(sp,lang_tool.check(sp)).strip()
            gr=_clean(gr)
        except: gr=sp
    else: gr=sp
    v5=gr[0].upper()+gr[1:] if gr else gr
    v5=_re.sub(r'(\.\s+)([a-z])',lambda m:m.group(1)+m.group(2).upper(),v5)
    if v5 and v5[-1] not in '.!?': v5+='.'
    return [sp.title(),(gr[0].upper()+gr[1:].lower()) if gr else gr,
            gr.upper(),gr.lower(),v5]

def _ac_thread(txt):
    global ac_busy
    try:
        sb._s('⏳ Correcting …')
        sb.suggs=_correct(txt); sb.show_ac=True
        sb._s('✓ Done — pick 1/2/3/4/5')
    except Exception as e:
        sb._s(f'Error: {str(e)[:40]}')
    finally:
        ac_busy=False; _notify_state()

def trigger_ac():
    global ac_busy
    if ac_busy: sb._s('Already correcting …'); return
    t=sb.live().strip()
    if not t: sb._s('Nothing to correct'); return
    if not _tools_event.is_set(): sb._s('⏳ Tools loading …'); return
    ac_busy=True
    threading.Thread(target=_ac_thread,args=(t,),daemon=True).start()

# ════════════════════════════════════════════════════════
# TRANSLATION ENGINE
# ════════════════════════════════════════════════════════
LANGUAGES = [
    {"code":"hi",    "bcp":"hi-IN",  "name":"Hindi",      "flag":"🇮🇳","native":"हिंदी"},
    {"code":"gu",    "bcp":"gu-IN",  "name":"Gujarati",   "flag":"🇮🇳","native":"ગુજરાતી"},
    {"code":"es",    "bcp":"es-ES",  "name":"Spanish",    "flag":"🇪🇸","native":"Español"},
    {"code":"fr",    "bcp":"fr-FR",  "name":"French",     "flag":"🇫🇷","native":"Français"},
    {"code":"de",    "bcp":"de-DE",  "name":"German",     "flag":"🇩🇪","native":"Deutsch"},
    {"code":"pt",    "bcp":"pt-BR",  "name":"Portuguese", "flag":"🇧🇷","native":"Português"},
    {"code":"ar",    "bcp":"ar-SA",  "name":"Arabic",     "flag":"🇸🇦","native":"العربية"},
    {"code":"zh-CN", "bcp":"zh-CN",  "name":"Chinese",    "flag":"🇨🇳","native":"中文"},
    {"code":"ja",    "bcp":"ja-JP",  "name":"Japanese",   "flag":"🇯🇵","native":"日本語"},
    {"code":"ko",    "bcp":"ko-KR",  "name":"Korean",     "flag":"🇰🇷","native":"한국어"},
    {"code":"ru",    "bcp":"ru-RU",  "name":"Russian",    "flag":"🇷🇺","native":"Русский"},
    {"code":"it",    "bcp":"it-IT",  "name":"Italian",    "flag":"🇮🇹","native":"Italiano"},
    {"code":"nl",    "bcp":"nl-NL",  "name":"Dutch",      "flag":"🇳🇱","native":"Nederlands"},
    {"code":"tr",    "bcp":"tr-TR",  "name":"Turkish",    "flag":"🇹🇷","native":"Türkçe"},
    {"code":"bn",    "bcp":"bn-BD",  "name":"Bengali",    "flag":"🇧🇩","native":"বাংলা"},
    {"code":"ur",    "bcp":"ur-PK",  "name":"Urdu",       "flag":"🇵🇰","native":"اردو"},
    {"code":"ta",    "bcp":"ta-IN",  "name":"Tamil",      "flag":"🇮🇳","native":"தமிழ்"},
]

_translation={"text":"","lang_code":"","lang_bcp":"","lang_name":"","flag":""}
_translate_busy=False

def _do_translate(text:str,lang_code:str)->str:
    try:
        from deep_translator import GoogleTranslator
        result=GoogleTranslator(source="auto",target=lang_code).translate(text)
        return result or text
    except Exception as e:
        print(f"[TRANSLATE] Error: {e}")
        return f"[Translation error: {e}]"

def _translate_thread(text:str,lang:dict):
    global _translate_busy,_translation
    try:
        sb._s(f"⏳ Translating to {lang['name']} …")
        translated=_do_translate(text,lang["code"])
        _translation={"text":translated,"lang_code":lang["code"],
                      "lang_bcp":lang["bcp"],"lang_name":lang["name"],"flag":lang["flag"]}
        sb._s(f"✓ Translated to {lang['name']} — click 🔊 to speak")
    except Exception as e:
        sb._s(f"Translation failed: {str(e)[:40]}")
    finally:
        _translate_busy=False; _notify_state()

# ════════════════════════════════════════════════════════
# STATE NOTIFICATION
# ════════════════════════════════════════════════════════
_state_event=threading.Event()

def _notify_state(): _state_event.set()

def _get_state_dict()->dict:
    return {
        "live_text":               sb.live(),
        "history":                 sb.sents[-10:],
        "word_suggestions":        word_suggs[:5],
        "autocorrect_suggestions": sb.suggs[:5],
        "show_autocorrect":        sb.show_ac,
        "status":                  sb.status,
        "model_ready":             _model_ready,
        "tools_ready":             _tools_ready,
        "translation":             _translation,
        "languages":               LANGUAGES,
    }

# ════════════════════════════════════════════════════════
# DRAWING HELPERS
# ════════════════════════════════════════════════════════
def skeleton(frame,res):
    mpd.draw_landmarks(frame,res.multi_hand_landmarks[0],mph.HAND_CONNECTIONS,
                       mps.get_default_hand_landmarks_style(),
                       mps.get_default_hand_connections_style())

def arc(frame,cx,cy,p,r=30,cf=(0,255,100),cp=(0,210,255)):
    cv2.circle(frame,(cx,cy),r,(40,40,40),2)
    if p>0:
        col=cf if p>=1.0 else cp
        cv2.ellipse(frame,(cx,cy),(r,r),-90,0,int(360*p),col,3)
        txt=f'{int(p*100)}%'; tw=cv2.getTextSize(txt,FS,0.40,1)[0][0]
        cv2.putText(frame,txt,(cx-tw//2,cy+5),FS,0.40,col,1,cv2.LINE_AA)

def hand_box(frame,lbl,conf,bx,by,bw,bh):
    H,W=frame.shape[:2]
    if lbl=='Uncertain': col=(30,30,180)
    elif lbl in SPECIALS: col=(0,150,255)
    else: g=int(255*min(conf,1.0)); col=(0,g,255-g)
    x1,y1=max(0,bx-12),max(0,by-12); x2,y2=min(W,bx+bw+12),min(H,by+bh+12)
    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
    txt=f'{lbl}  {conf:.0%}' if lbl!='Uncertain' else 'Uncertain'
    (tw,th),_=cv2.getTextSize(txt,FD,0.90,2); ty=max(th+12,y1-6)
    cv2.rectangle(frame,(x1,ty-th-6),(x1+tw+14,ty+6),col,-1)
    cv2.putText(frame,txt,(x1+7,ty),FD,0.90,WHITE,2,cv2.LINE_AA)

def _rect(frame,x1,y1,x2,y2,col,alpha=1.0):
    if alpha<1.0:
        ov=frame.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
        cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    else: cv2.rectangle(frame,(x1,y1),(x2,y2),col,-1)

def draw_top(frame,probs):
    W=frame.shape[1]; cv2.rectangle(frame,(0,0),(W,TOP_H),DARK,-1)
    cv2.line(frame,(0,TOP_H),(W,TOP_H),SEP,1); CONF_W=180
    cv2.putText(frame,'History :',(12,22),FD,0.65,(200,100,255),2,cv2.LINE_AA)
    hist=sb.history(3)
    for i,s in enumerate(hist if hist else []):
        disp=s if len(s)<=62 else '...'+s[-60:]
        cv2.putText(frame,disp,(12,44+i*22),FS,0.52,(170,80,220),1,cv2.LINE_AA)
    if not hist: cv2.putText(frame,'(no sentences yet)',(12,44),FS,0.46,(80,50,100),1,cv2.LINE_AA)
    cv2.putText(frame,'Live Feed :',(12,TOP_H-12),FD,0.68,(60,60,230),2,cv2.LINE_AA)
    lf_lw=cv2.getTextSize('Live Feed :',FD,0.68,2)[0][0]+12
    live_text=sb.live()
    if live_text:
        cursor=live_text+chr(9608)
        if len(cursor)>52: cursor='...'+cursor[-50:]
        cv2.putText(frame,cursor,(12+lf_lw,TOP_H-12),FD,0.78,(60,60,230),2,cv2.LINE_AA)
    if probs is not None:
        cv2.line(frame,(W-CONF_W,2),(W-CONF_W,TOP_H-2),SEP,1)
        for i,idx in enumerate(np.argsort(probs)[::-1][:3]):
            lbl=class_names[idx]; p=probs[idx]; bw=int(p*(CONF_W-55))
            y0=8+i*32; x0=W-CONF_W+6
            col=(0,120,200) if lbl in SPECIALS else (35,130,35)
            cv2.rectangle(frame,(x0,y0),(x0+CONF_W-58,y0+22),(30,30,30),-1)
            cv2.rectangle(frame,(x0,y0),(x0+bw,y0+22),col,-1)
            cv2.putText(frame,f'{lbl} {p:.0%}',(x0+4,y0+15),FS,0.40,WHITE,1,cv2.LINE_AA)

def draw_bottom(frame):
    H,W=frame.shape[:2]; has_w=bool(word_suggs); has_a=sb.show_ac and bool(sb.suggs)
    ph=50; ph+=40 if has_w else 0; ph+=10 if has_w else 0
    ph+=150 if has_a else 0; ph+=8 if has_a else 0
    py=H-ph; _rect(frame,0,py,W,H,DARK,alpha=0.92)
    cv2.line(frame,(0,py),(W,py),SEP,1); y=py+18
    if time.time()-sb.st<4.0:
        cv2.circle(frame,(10,y-4),4,TEAL,-1)
        cv2.putText(frame,sb.status,(20,y),FS,0.48,CYAN,1,cv2.LINE_AA)
    y+=18
    cv2.putText(frame,'FLAT=spc  PINCH=end+speak  THUMB=correct  Q-T=word  1-5=fix  BKSP=del  C=clr',
                (10,y),FS,0.32,(52,52,52),1,cv2.LINE_AA)
    y+=8
    if has_w:
        y+=6; cv2.line(frame,(0,y),(W,y),(38,38,38),1); y+=5
        cv2.putText(frame,'Suggestions',(10,y+17),FS,0.38,(70,70,70),1,cv2.LINE_AA)
        LABEL_W=108; avail=W-LABEL_W-8; CW=avail//5; CH=26
        KEYS=['Q','W','E','R','T']; KCOLS=[GREEN,BLUE,ORANGE,PINK,PURPLE]
        for i,word in enumerate(word_suggs[:5]):
            cx1=LABEL_W+i*CW+2; cx2=cx1+CW-4; cy1=y+2; cy2=y+CH
            cv2.rectangle(frame,(cx1,cy1),(cx2,cy2),CHIP_BG,-1)
            cv2.rectangle(frame,(cx1,cy1),(cx2,cy2),CHIP_BOR,1)
            cv2.circle(frame,(cx1+11,cy1+13),9,KCOLS[i],-1)
            cv2.putText(frame,KEYS[i],(cx1+7,cy1+17),FS,0.32,BLACK_C,1,cv2.LINE_AA)
            mx=max(3,(CW-28)//8); dw=word if len(word)<=mx else word[:mx-1]+'…'
            cv2.putText(frame,dw,(cx1+24,cy1+17),FS,0.42,CHIP_TXT,1,cv2.LINE_AA)
        y+=CH+6
    if has_a:
        y+=6; cv2.line(frame,(0,y),(W,y),(38,38,38),1); y+=8
        cv2.putText(frame,'Correction — press 1/2/3/4/5:',(12,y+13),FS,0.44,YELLOW,1,cv2.LINE_AA)
        y+=16; ACOLS=[GREEN,BLUE,ORANGE,PINK,PURPLE]
        ALABELS=['Title Case','Sentence  ','ALL CAPS  ','lowercase ','Polished  ']
        for i,opt in enumerate(sb.suggs[:5]):
            cv2.rectangle(frame,(8,y+i*27+3),(14,y+i*27+22),ACOLS[i],-1)
            disp=f'[{i+1}] {ALABELS[i]}: {opt}'
            if len(disp)>76: disp=disp[:73]+'…'
            cv2.putText(frame,disp,(20,y+i*27+17),FS,0.44,ACOLS[i],1,cv2.LINE_AA)

# ════════════════════════════════════════════════════════
# PROCESS A SINGLE FRAME  (called from /ws/camera)
# Browser sends JPEG bytes → server annotates → returns JPEG bytes
# ════════════════════════════════════════════════════════
def process_frame(jpg_bytes:bytes)->bytes:
    nparr=np.frombuffer(jpg_bytes,np.uint8)
    frame=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    if frame is None: return jpg_bytes

    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=detector.process(rgb); probs_hud=None

    if res.multi_hand_landmarks:
        skeleton(frame,res)
        vec=extract_landmarks(res)
        if vec is not None and model is not None:
            probs=model.predict(vec.reshape(1,-1),verbose=0)[0]
            smoother.push(probs); idx,conf=smoother.top(); probs_hud=probs
            lbl=class_names[idx] if conf>=CONF_THRESH else 'Uncertain'
            lms=res.multi_hand_landmarks[0].landmark
            fH,fW=frame.shape[:2]
            xs=[int(l.x*fW) for l in lms]; ys=[int(l.y*fH) for l in lms]
            bx,by=min(xs),min(ys); bw,bh=max(xs)-bx,max(ys)-by
            cx,cy=bx+bw//2,max(55,by-60)
            hand_box(frame,lbl,conf,bx,by,bw,bh)
            with _lock:
                if lbl in SPECIALS:
                    fired=hold.update(lbl)
                    arc(frame,cx,cy,hold.prog(),r=36,cf=(0,255,100),cp=(0,165,255))
                    if fired:
                        if   lbl==CLS_SPACE: sb.space()
                        elif lbl==CLS_DOT:   sb.dot()
                        elif lbl==CLS_AC:    trigger_ac()
                elif lbl!='Uncertain':
                    hold.update(lbl)
                    cap_lbl=lt.update(lbl); p=lt.prog()
                    arc(frame,cx,cy,p,r=26,cf=(0,255,100),cp=(0,215,215))
                    if cap_lbl: sb.add(cap_lbl); print(f"  [{cap_lbl}] {sb.cw}")
    else:
        smoother.reset(); hold.update(None); lt.reset()
        H,W=frame.shape[:2]; msg='No hand detected'
        tw=cv2.getTextSize(msg,FS,0.75,2)[0][0]
        cv2.putText(frame,msg,((W-tw)//2,H//2),FS,0.75,(55,55,200),2,cv2.LINE_AA)

    draw_top(frame,probs_hud); draw_bottom(frame)
    ok,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
    return buf.tobytes() if ok else jpg_bytes

# ════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════
async def _state_broadcaster():
    loop=asyncio.get_event_loop(); last={}
    while True:
        await loop.run_in_executor(None,_state_event.wait,0.05)
        _state_event.clear(); current=_get_state_dict()
        if current!=last:
            await ws_manager.broadcast(current); last=dict(current)

async def _tts_broadcaster():
    loop=asyncio.get_event_loop()
    while True:
        try:
            msg=await loop.run_in_executor(None,_tts_audio_queue.get,True,0.1)
            await ws_manager.broadcast(msg)
        except queue.Empty: pass
        except Exception: pass

@asynccontextmanager
async def lifespan(application:FastAPI):
    asyncio.create_task(_state_broadcaster())
    asyncio.create_task(_tts_broadcaster())
    yield

app.router.lifespan_context=lifespan

# ════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════

@app.websocket("/ws/camera")
async def camera_ws(ws:WebSocket):
    """Browser sends JPEG frames as binary, server returns annotated JPEG as binary."""
    await ws.accept()
    try:
        while True:
            data=await ws.receive_bytes()
            loop=asyncio.get_event_loop()
            result_jpg=await loop.run_in_executor(None,process_frame,data)
            await ws.send_bytes(result_jpg)
    except WebSocketDisconnect: pass
    except Exception as e: print(f"[camera_ws] {e}")

@app.websocket("/ws")
async def websocket_endpoint(ws:WebSocket):
    await ws_manager.connect(ws)
    await ws.send_text(json.dumps(_get_state_dict()))
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(ws)

class Cmd(BaseModel):
    action:str; index:Optional[int]=0

@app.post("/command")
def command(cmd:Cmd):
    with _lock:
        a=cmd.action; i=cmd.index or 0
        if   a=="space":       sb.space()
        elif a=="dot":         sb.dot()
        elif a=="autocorrect": trigger_ac()
        elif a=="backspace":   sb.backspace()
        elif a=="clear":       sb.clear(); update_suggs()
        elif a=="pick_word":   apply_sugg(i)
        elif a=="pick_ac":     sb.pick(i)
        else: return {"ok":False,"error":f"Unknown: {a}"}
    _notify_state(); return {"ok":True}

@app.get("/health")
def health():
    return {"status":"ok","model_ready":_model_ready,"tools_ready":_tools_ready}

class TranslateReq(BaseModel):
    text:str; lang_code:str

@app.post("/translate")
def translate_endpoint(req:TranslateReq):
    global _translate_busy,_translation
    if not req.text.strip(): return {"ok":False,"error":"No text to translate"}
    if _translate_busy: return {"ok":False,"error":"Translation in progress"}
    lang=next((l for l in LANGUAGES if l["code"]==req.lang_code),None)
    if not lang: return {"ok":False,"error":f"Unknown language: {req.lang_code}"}
    _translate_busy=True
    _translation={"text":"","lang_code":"","lang_bcp":"","lang_name":"","flag":""}
    threading.Thread(target=_translate_thread,args=(req.text,lang),daemon=True).start()
    return {"ok":True}

@app.get("/languages")
def get_languages(): return {"languages":LANGUAGES}

def _gtts_lang_code(bcp:str)->str:
    mapping={"hi-IN":"hi","gu-IN":"gu","es-ES":"es","fr-FR":"fr","de-DE":"de",
             "pt-BR":"pt","ar-SA":"ar","zh-CN":"zh-CN","ja-JP":"ja","ko-KR":"ko",
             "ru-RU":"ru","it-IT":"it","nl-NL":"nl","tr-TR":"tr","bn-BD":"bn",
             "ur-PK":"ur","ta-IN":"ta"}
    return mapping.get(bcp,bcp.split("-")[0])

class SpeakTranslationReq(BaseModel):
    text:str; lang_bcp:str

@app.post("/speak_translation")
def speak_translation_endpoint(req:SpeakTranslationReq):
    if not req.text.strip(): return {"ok":False,"error":"No text to speak"}
    lang_code=_gtts_lang_code(req.lang_bcp)
    threading.Thread(target=lambda:_queue_speak(req.text,lang=lang_code),daemon=True).start()
    return {"ok":True}

# ════════════════════════════════════════════════════════
# HTML UI
# ════════════════════════════════════════════════════════
_HTML="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SignatriX AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#F0F4FF;--bg2:#FFFFFF;--bg3:#F7F9FF;
  --border:#E2E8F4;--border2:#CBD5E8;
  --text:#0F172A;--text2:#334155;--text3:#64748B;--text4:#94A3B8;
  --blue:#3B82F6;--blue2:#1D4ED8;--blue-l:#EFF6FF;
  --purple:#7C3AED;--purple-l:#F5F3FF;
  --green:#10B981;--green-l:#ECFDF5;
  --orange:#F59E0B;--orange-l:#FFFBEB;
  --red:#EF4444;--red-l:#FEF2F2;
  --pink:#EC4899;--teal:#14B8A6;
  --sh:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
  --sh-md:0 4px 12px rgba(0,0,0,.08),0 2px 4px rgba(0,0,0,.04);
  --sh-lg:0 10px 24px rgba(0,0,0,.09),0 4px 6px rgba(0,0,0,.04);
  --sh-xl:0 24px 40px rgba(0,0,0,.12),0 8px 12px rgba(0,0,0,.05);
  --r:16px;--r2:11px;--r3:8px;
  --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);overflow-x:hidden}
body{background-image:radial-gradient(circle,#C7D2FE 1px,transparent 1px);background-size:28px 28px;background-attachment:fixed}
.app{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column}
nav{display:flex;align-items:center;justify-content:space-between;padding:16px 56px;
  background:rgba(255,255,255,.94);backdrop-filter:blur(20px);
  border-bottom:1.5px solid var(--border);position:sticky;top:0;z-index:100;box-shadow:var(--sh)}
.logo{font-family:var(--display);font-weight:800;font-size:1.4rem;letter-spacing:-.02em;
  background:linear-gradient(135deg,var(--blue),var(--purple) 55%,var(--pink));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.nav-r{display:flex;align-items:center;gap:10px}
.badge{font-family:var(--mono);font-size:.72rem;letter-spacing:.07em;padding:6px 14px;
  border-radius:20px;border:1.5px solid;font-weight:500;transition:all .3s}
.badge.on{color:#059669;border-color:#A7F3D0;background:#ECFDF5;animation:glow 2.2s ease-in-out infinite}
.badge.off{color:var(--red);border-color:#FECACA;background:var(--red-l)}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.2)}50%{box-shadow:0 0 0 7px rgba(16,185,129,0)}}
.nav-btn{font-size:.85rem;font-weight:500;padding:9px 20px;border-radius:var(--r3);
  border:1.5px solid var(--border2);background:var(--bg2);color:var(--text2);
  cursor:pointer;transition:all .18s;box-shadow:var(--sh);font-family:var(--font)}
.nav-btn:hover{background:var(--blue-l);border-color:var(--blue);color:var(--blue);transform:translateY(-1px)}
.hero{text-align:center;padding:24px 24px 16px}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.18em;color:var(--purple);
  text-transform:uppercase;margin-bottom:14px;display:inline-flex;align-items:center;gap:10px}
.eyebrow::before,.eyebrow::after{content:'';width:32px;height:1.5px;background:var(--purple);opacity:.35}
.title{font-family:var(--display);font-weight:800;font-size:clamp(2rem,4vw,3.2rem);line-height:1;letter-spacing:-.03em;margin-bottom:10px}
.w1{color:var(--text)}.w2{background:linear-gradient(135deg,var(--blue),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.w3{background:linear-gradient(135deg,var(--pink),var(--red));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sub{font-size:.95rem;color:var(--text3);max-width:540px;margin:0 auto 12px;line-height:1.6}
.pills{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
.pill{font-family:var(--mono);font-size:.70rem;font-weight:500;padding:5px 13px;border-radius:20px;border:1.5px solid;letter-spacing:.04em}
.pl1{color:var(--blue2);border-color:#BFDBFE;background:var(--blue-l)}
.pl2{color:var(--purple);border-color:#DDD6FE;background:var(--purple-l)}
.pl3{color:var(--green);border-color:#A7F3D0;background:var(--green-l)}
.pl4{color:#92400E;border-color:#FDE68A;background:var(--orange-l)}
.grid{display:grid;grid-template-columns:1fr 360px;gap:24px;padding:0 40px 32px;flex:1;align-items:start;min-width:0}
@media(max-width:1100px){.grid{grid-template-columns:1fr;padding:0 16px 24px}}
.sidebar{display:flex;flex-direction:column;gap:12px;min-width:0;width:360px}
.sidebar .card{margin-bottom:0;width:100%;min-width:0;box-sizing:border-box}
.cam-wrap{background:var(--bg2);border:1.5px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--sh-lg);min-width:0;width:100%}
.cam-bar{display:flex;align-items:center;gap:8px;padding:14px 22px;border-bottom:1.5px solid var(--border);background:var(--bg3)}
.dot{width:11px;height:11px;border-radius:50%}
.dot.r{background:#FF5F57}.dot.a{background:#FEBC2E}
.dot.g{background:#28C840;animation:glow 1.8s ease-in-out infinite}
.cam-ttl{font-family:var(--mono);font-size:.68rem;color:var(--text4);letter-spacing:.09em;margin-left:6px}
.cam-body{position:relative;background:#0F172A;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;overflow:hidden}
#cam-canvas{width:100%;height:100%;object-fit:cover;display:none}
.cam-perm{display:flex;flex-direction:column;align-items:center;gap:14px;padding:48px 24px;text-align:center}
.cam-perm p{font-size:.88rem;color:#64748B;line-height:1.8}
.start-btn{padding:12px 28px;background:var(--blue);color:#fff;border:none;border-radius:var(--r3);
  font-weight:600;font-size:.9rem;cursor:pointer;transition:all .18s;box-shadow:var(--sh-md)}
.start-btn:hover{background:var(--blue2);transform:translateY(-2px);box-shadow:var(--sh-lg)}
.ctrl-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:12px 16px 8px}
.ctrl{font-family:var(--mono);font-size:.65rem;padding:6px 8px;border-radius:7px;background:var(--bg3);border:1.5px solid var(--border);color:var(--text3);text-align:center;line-height:1.4}
.ctrl b{color:var(--blue);font-weight:600;display:block;margin-bottom:2px}
.cmd-row{display:flex;gap:8px;padding:8px 16px 16px;flex-wrap:wrap}
.cmd{font-family:var(--font);font-size:.82rem;font-weight:500;padding:10px 14px;border-radius:var(--r3);
  border:1.5px solid var(--border2);background:var(--bg2);color:var(--text2);
  cursor:pointer;transition:all .15s;flex:1;min-width:70px;text-align:center;box-shadow:var(--sh)}
.cmd:hover{transform:translateY(-2px);box-shadow:var(--sh-md)}.cmd:active{transform:scale(.97)}
.cmd.blue:hover{background:var(--blue-l);border-color:var(--blue);color:var(--blue)}
.cmd.green:hover{background:var(--green-l);border-color:var(--green);color:var(--green)}
.cmd.purple:hover{background:var(--purple-l);border-color:var(--purple);color:var(--purple)}
.cmd.orange:hover{background:var(--orange-l);border-color:var(--orange);color:var(--orange)}
.cmd.red:hover{background:var(--red-l);border-color:var(--red);color:var(--red)}
.card{background:var(--bg2);border:1.5px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--sh);transition:box-shadow .2s;min-width:0;width:100%}
.card:hover{box-shadow:var(--sh-md)}
.card-hdr{display:flex;align-items:center;gap:10px;padding:11px 18px;border-bottom:1.5px solid var(--border);background:var(--bg3)}
.hdr-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.hdr-lbl{font-size:.76rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--text3)}
.hdr-tag{margin-left:auto;font-family:var(--mono);font-size:.63rem;font-weight:500;padding:3px 9px;border-radius:10px;border:1.5px solid}
.card-body{padding:14px 16px;min-width:0;overflow:hidden}
.live{font-family:var(--mono);font-size:1.22rem;font-weight:500;color:var(--text);min-height:48px;line-height:1.6;word-break:break-all}
.cur{display:inline-block;width:10px;height:1.1em;background:var(--blue);vertical-align:text-bottom;border-radius:2px;animation:blink 1s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.live-ph{color:var(--text4);font-size:.80rem;letter-spacing:.04em}
.sbar{display:flex;align-items:center;gap:9px;padding:11px 16px;border-radius:var(--r2);margin-bottom:0;font-size:.82rem;font-weight:500;border:1.5px solid;min-width:0;width:100%;box-sizing:border-box}
.sbar.info{color:var(--blue2);border-color:#BFDBFE;background:var(--blue-l)}
.sbar.ok{color:#065F46;border-color:#A7F3D0;background:var(--green-l)}
.sbar.warn{color:#92400E;border-color:#FDE68A;background:var(--orange-l)}
.hi{font-size:.90rem;color:var(--text2);padding:10px 14px;border-radius:var(--r3);border-left:3.5px solid var(--purple);background:var(--purple-l);margin-bottom:9px;line-height:1.55;transition:all .18s}
.hi:hover{background:#EDE9FE;transform:translateX(4px)}
.hi-ph{font-family:var(--mono);font-size:.72rem;color:var(--text4);text-align:center;padding:6px 0}
.sr{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:7px}
.sc{font-family:var(--mono);font-size:.74rem;font-weight:500;padding:7px 14px;border-radius:20px;border:1.5px solid;cursor:pointer;transition:all .15s;display:flex;align-items:center;gap:6px}
.sc:hover{transform:translateY(-2px);box-shadow:var(--sh-md)}.sc:active{transform:scale(.96)}
.sc.s0{color:#065F46;border-color:#A7F3D0;background:var(--green-l)}
.sc.s1{color:var(--blue2);border-color:#BFDBFE;background:var(--blue-l)}
.sc.s2{color:#92400E;border-color:#FDE68A;background:var(--orange-l)}
.sc.s3{color:#831843;border-color:#FBCFE8;background:#FDF2F8}
.sc.s4{color:#4C1D95;border-color:#DDD6FE;background:var(--purple-l)}
.skey{font-size:.62rem;opacity:.55;background:rgba(0,0,0,.07);padding:2px 6px;border-radius:4px}
.sh{font-family:var(--mono);font-size:.64rem;color:var(--text4);margin-top:6px}
.ao{display:flex;align-items:flex-start;gap:12px;padding:11px 14px;border-radius:var(--r3);margin-bottom:7px;cursor:pointer;border:1.5px solid var(--border);background:var(--bg3);transition:all .15s}
.ao:hover{background:var(--blue-l);border-color:#BFDBFE;transform:translateX(4px)}
.an{font-family:var(--mono);font-size:.68rem;font-weight:700;min-width:24px;padding-top:2px}
.al{font-family:var(--mono);font-size:.64rem;color:var(--text4);text-transform:uppercase;letter-spacing:.07em;min-width:72px;padding-top:2px}
.av{font-size:.86rem;color:var(--text2);line-height:1.5;font-weight:500}
.ac0{color:var(--green)}.ac1{color:var(--blue)}.ac2{color:var(--orange)}.ac3{color:var(--pink)}.ac4{color:var(--purple)}
.str{display:flex;align-items:center;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--border)}
.str:last-child{border-bottom:none}
.stk{font-family:var(--mono);font-size:.68rem;color:var(--text3);letter-spacing:.07em}
.stv{font-family:var(--mono);font-size:.68rem;font-weight:600;display:flex;align-items:center;gap:6px}
.sd{width:8px;height:8px;border-radius:50%}
.modal-bg{position:fixed;inset:0;z-index:999;background:rgba(15,23,42,.50);backdrop-filter:blur(14px);display:none;align-items:flex-start;justify-content:center;padding:36px 16px;overflow-y:auto}
.modal-bg.open{display:flex}
.modal{background:var(--bg2);border:1.5px solid var(--border);border-radius:22px;width:min(720px,96vw);box-shadow:var(--sh-xl)}
.modal-top{padding:30px 36px 22px;border-bottom:1.5px solid var(--border);display:flex;align-items:flex-start;justify-content:space-between;background:var(--bg3);border-radius:22px 22px 0 0}
.mt{font-family:var(--display);font-weight:800;font-size:1.6rem;color:var(--text)}
.ms{font-size:.84rem;color:var(--text3);margin-top:5px}
.mc{width:36px;height:36px;border-radius:50%;border:1.5px solid var(--border2);background:var(--bg2);color:var(--text3);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:.95rem;font-weight:700;transition:all .18s}
.mc:hover{background:var(--red-l);border-color:#FECACA;color:var(--red)}
.modal-body{padding:30px 36px 36px}
.msec{font-family:var(--display);font-weight:700;font-size:.80rem;letter-spacing:.12em;text-transform:uppercase;color:var(--blue2);margin-bottom:14px;margin-top:24px;display:flex;align-items:center;gap:10px}
.msec:first-child{margin-top:0}.msec::after{content:'';flex:1;height:1.5px;background:var(--border)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-family:var(--mono);font-size:.72rem;font-weight:500;padding:6px 14px;border-radius:20px;border:1.5px solid;letter-spacing:.04em}
.cb{color:var(--blue2);border-color:#BFDBFE;background:var(--blue-l)}
.cp{color:var(--purple);border-color:#DDD6FE;background:var(--purple-l)}
.cg{color:#065F46;border-color:#A7F3D0;background:var(--green-l)}
.co{color:#92400E;border-color:#FDE68A;background:var(--orange-l)}
.ck{color:#831843;border-color:#FBCFE8;background:#FDF2F8}
.ct{color:#0F766E;border-color:#99F6E4;background:#F0FDFA}
.lang-grid{display:flex;flex-direction:row;flex-wrap:wrap;gap:5px;margin-bottom:12px;min-width:0}
.lang-btn{display:inline-flex;align-items:center;gap:4px;padding:4px 8px;border-radius:20px;border:1.5px solid var(--border);background:var(--bg3);cursor:pointer;transition:all .16s;font-family:var(--font);white-space:nowrap;flex-shrink:1;line-height:1;min-width:0}
.lang-btn:hover{border-color:var(--teal);background:#F0FDFA;transform:translateY(-1px)}
.lang-btn.active{border-color:var(--teal);background:#CCFBF1;color:#0F766E;font-weight:600}
.lang-flag{font-size:.85rem;line-height:1}.lang-name{font-size:.70rem;font-weight:500;color:var(--text2)}
.trans-result{font-size:1.10rem;font-weight:500;color:var(--text);min-height:44px;line-height:1.6;word-break:break-word;margin-bottom:12px;font-family:var(--font);padding:10px 14px;background:var(--bg3);border:1.5px solid var(--border);border-radius:var(--r3)}
.trans-result.empty{color:var(--text4);font-size:.80rem;font-style:italic}
.speak-btn{width:100%;padding:10px;border-radius:var(--r3);border:1.5px solid #99F6E4;background:#F0FDFA;color:#0F766E;font-weight:600;font-size:.84rem;cursor:pointer;transition:all .16s;display:flex;align-items:center;justify-content:center;gap:7px}
.speak-btn:hover{background:#CCFBF1;transform:translateY(-1px)}.speak-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
footer{border-top:1.5px solid var(--border);padding:20px 52px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.92);backdrop-filter:blur(12px)}
.fm{font-size:.87rem;color:var(--text3);font-weight:500}.fm .heart{color:var(--pink)}.fm .name{color:var(--purple);font-weight:700}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg3)}::-webkit-scrollbar-thumb{background:#CBD5E8;border-radius:3px}

/* ===== MODEL INFO SECTION ===== */
.model-info{padding:56px 40px 64px;background:linear-gradient(180deg,#F7F9FF 0%,#ffffff 100%);border-top:1.5px solid var(--border);margin-top:0}
.mi-header{text-align:center;margin-bottom:48px}
.mi-eyebrow{font-family:var(--mono);font-size:.70rem;letter-spacing:.18em;color:var(--purple);text-transform:uppercase;margin-bottom:12px;display:inline-flex;align-items:center;gap:10px}
.mi-eyebrow::before,.mi-eyebrow::after{content:'';width:28px;height:1.5px;background:var(--purple);opacity:.35}
.section-title{font-family:var(--display);font-size:clamp(1.6rem,3vw,2.2rem);font-weight:800;letter-spacing:-.03em;color:var(--text);margin-bottom:10px}
.section-sub{font-size:.88rem;color:var(--text3);max-width:480px;margin:0 auto}
.mi-grid{display:grid;grid-template-columns:1fr 1fr;gap:28px;max-width:1100px;margin:0 auto}
@media(max-width:860px){.mi-grid{grid-template-columns:1fr}}
.mi-col{display:flex;flex-direction:column;gap:18px}
.info-card{background:#ffffff;padding:22px 24px;border-radius:var(--r);box-shadow:var(--sh-md);border:1.5px solid var(--border);transition:box-shadow .2s}
.info-card:hover{box-shadow:var(--sh-lg)}
.info-card-hdr{display:flex;align-items:center;gap:10px;margin-bottom:14px;padding-bottom:12px;border-bottom:1.5px solid var(--border)}
.info-card-icon{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0}
.info-card h3{font-family:var(--display);font-size:.88rem;font-weight:700;letter-spacing:-.01em;color:var(--text)}
.info-list{list-style:none;display:flex;flex-direction:column;gap:8px}
.info-list li{display:flex;align-items:center;gap:9px;font-size:.84rem;color:var(--text2);line-height:1.4}
.info-list li::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--blue);flex-shrink:0}
.status-rows{display:flex;flex-direction:column;gap:0}
.status-row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--border)}
.status-row:last-child{border-bottom:none}
.status-row span:first-child{font-family:var(--mono);font-size:.70rem;color:var(--text3);letter-spacing:.06em;text-transform:uppercase}
.image-card{background:#ffffff;border-radius:var(--r);box-shadow:var(--sh-md);border:1.5px solid var(--border);overflow:hidden;transition:box-shadow .2s}
.image-card:hover{box-shadow:var(--sh-lg)}
.image-card-hdr{display:flex;align-items:center;gap:10px;padding:16px 20px;border-bottom:1.5px solid var(--border);background:var(--bg3)}
.image-card-hdr span{font-size:.76rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--text3)}
.image-card img{width:100%;display:block;padding:16px}

</style>
</head>
<body>
<div class="app">

<nav>
  <div class="logo">🤟 SignatriX AI</div>
  <div class="nav-r">
    <div class="badge off" id="ws-badge">○ Connecting …</div>
    <button class="nav-btn" onclick="openModal()">📋 Model Info</button>
  </div>
</nav>

<div class="hero">
  <div class="eyebrow">Empowering Communication Without Limits</div>
  <div class="title">
  <span class="w1">SignatriX</span>
  <span class="w2">AI</span>
</div>
  <p class="sub">American Sign Language → Text → Speech · Multilingual Support<br/>MobileNetV2 · MediaPipe · WebSocket · Browser Camera · Experiment-Ready</p>
  <div class="pills">
    <div class="pill pl1">MobileNetV2</div>
    <div class="pill pl2">MediaPipe Hands</div>
    <div class="pill pl3"> WebSocket</div>
    <div class="pill pl4">Browser Camera API</div>
  </div>
</div>

<div class="grid">
  <div style="min-width:0;width:100%">
    <div class="cam-wrap">
      <div class="cam-bar">
        <div class="dot r"></div><div class="dot a"></div><div class="dot g"></div>
        <span class="cam-ttl">CAMERA_FEED · Browser → Server → Canvas · MobileNetV2 + MediaPipe</span>
      </div>
      <div class="cam-body" id="cam-body">
        <div class="cam-perm" id="cam-perm">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="1.2">
            <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
          </svg>
          <p>Click below to start your camera.<br/><span style="font-size:.75rem;color:var(--text4)">Your browser will ask for camera permission.</span></p>
          <button class="start-btn" onclick="startCamera()">📷 Start Camera</button>
        </div>
        <canvas id="cam-canvas"></canvas>
      </div>
    </div>
    <div class="ctrl-strip">
      <div class="ctrl"><b>FLAT HAND</b> space</div>
      <div class="ctrl"><b>PINCH</b> end+speak</div>
      <div class="ctrl"><b>THUMBS UP</b> autocorrect</div>
      <div class="ctrl"><b>Q–T</b> word suggest</div>
      <div class="ctrl"><b>1–5</b> pick correction</div>
      <div class="ctrl"><b>BKSP/Z</b> delete</div>
      <div class="ctrl"><b>C</b> clear all</div>
    </div>
    <div class="cmd-row">
      <button class="cmd blue"   onclick="send('backspace')">⌫ Delete</button>
      <button class="cmd green"  onclick="send('space')">␣ Space</button>
      <button class="cmd purple" onclick="send('dot')">● End</button>
      <button class="cmd orange" onclick="send('autocorrect')">✦ Fix</button>
      <button class="cmd red"    onclick="send('clear')">✕ Clear</button>
    </div>
  </div>

  <div class="sidebar">
    <div class="sbar info" id="sbar" style="display:none"><span>›</span><span id="sbar-txt"></span></div>

    <div class="card">
      <div class="card-hdr">
        <div class="hdr-dot" style="background:var(--red)"></div>
        <div class="hdr-lbl">Live Feed</div>
        <div class="hdr-tag" style="color:var(--red);border-color:#FECACA;background:var(--red-l)">LIVE</div>
      </div>
      <div class="card-body">
        <div class="live" id="live"><span class="live-ph">Start signing …</span><span class="cur"></span></div>
      </div>
    </div>

    <div class="card" id="sugg-card" style="display:none">
      <div class="card-hdr">
        <div class="hdr-dot" style="background:var(--green)"></div>
        <div class="hdr-lbl">Word Suggestions</div>
      </div>
      <div class="card-body">
        <div class="sr" id="sr"></div>
        <div class="sh">Click or press Q &nbsp;W &nbsp;E &nbsp;R &nbsp;T</div>
      </div>
    </div>

    <div class="card" id="ac-card" style="display:none">
      <div class="card-hdr">
        <div class="hdr-dot" style="background:var(--orange)"></div>
        <div class="hdr-lbl">Autocorrect</div>
        <div class="hdr-tag" style="color:var(--orange);border-color:#FDE68A;background:var(--orange-l)">5 variants</div>
      </div>
      <div class="card-body" id="ac-body"></div>
    </div>

    <div class="card">
      <div class="card-hdr">
        <div class="hdr-dot" style="background:var(--purple)"></div>
        <div class="hdr-lbl">History</div>
        <div class="hdr-tag" id="hc" style="color:var(--purple);border-color:#DDD6FE;background:var(--purple-l)">0 sentences</div>
      </div>
      <div class="card-body" id="hb"><div class="hi-ph">No sentences yet</div></div>
    </div>

    <div class="card">
      <div class="card-hdr">
        <div class="hdr-dot" style="background:var(--teal)"></div>
        <div class="hdr-lbl">Translate</div>
        <div class="hdr-tag" id="trans-lang-tag" style="color:var(--teal);border-color:#99F6E4;background:#F0FDFA">17 languages</div>
      </div>
      <div class="card-body">
        <div class="lang-grid" id="lang-grid"></div>
        <div class="trans-result empty" id="trans-result">Select a language to translate the last sentence …</div>
        <button class="speak-btn" id="speak-trans-btn" disabled onclick="speakTranslation()">🔊 Speak Translation</button>
      </div>
    </div>

    <div class="card">
      <div class="card-hdr">
        <div class="hdr-dot" style="background:var(--green)"></div>
        <div class="hdr-lbl">System Status</div>
      </div>
      <div class="card-body">
        <div class="str"><span class="stk">WEBSOCKET</span><span class="stv" id="st-ws"><span class="sd" style="background:var(--orange)"></span>Connecting …</span></div>
        <div class="str"><span class="stk">CAMERA</span><span class="stv" id="st-cam"><span class="sd" style="background:var(--text4)"></span>Not started</span></div>
        <div class="str"><span class="stk">ML MODEL</span><span class="stv" id="st-md"><span class="sd" style="background:var(--text4)"></span>—</span></div>
        <div class="str"><span class="stk">AUTOCORRECT</span><span class="stv" id="st-tl"><span class="sd" style="background:var(--text4)"></span>—</span></div>
        <div class="str"><span class="stk">TTS ENGINE</span><span class="stv" style="color:var(--green)"><span class="sd" style="background:var(--green)"></span>gTTS Cloud ✓</span></div>
      </div>
    </div>
  </div>
</div>

<footer><div class="fm">Made with <span class="heart">❤️</span> by <span class="name">Team SignatriX AI </span> &nbsp;·&nbsp; SignatriXAI &nbsp;·&nbsp; 27/02/2026</div></footer>
</div>

<!-- MODAL -->
<div class="modal-bg" id="modal">
 <div class="modal">
  <div class="modal-top">
   <div><div class="mt">🤟 SignatriX AI</div><div class="ms">experiment-Ready Edition — Browser Camera + gTTS</div></div>
   <button class="mc" onclick="closeModal()">✕</button>
  </div>
  <div class="modal-body">
   <div class="msec">⚙️ Tech Stack</div>
   <div class="chips">
    <div class="chip cb">Python 3.10</div><div class="chip co">TensorFlow / Keras</div>
    <div class="chip ct">MobileNetV2</div><div class="chip cb">MediaPipe Hands</div>
    <div class="chip cg">OpenCV Headless</div><div class="chip cp">FastAPI + Uvicorn</div>
    <div class="chip cp">WebSocket</div><div class="chip ck">gTTS (Cloud TTS)</div>
    <div class="chip cg">LanguageTool NLP</div><div class="chip cb">Browser Camera API</div>
   </div>
   <div class="msec">✋ Special Signs</div>
   <div class="chips">
    <div class="chip cb">✋ Flat Hand → SPACE (1.2s hold)</div>
    <div class="chip cp">🤏 Pinch → End Sentence + Speak</div>
    <div class="chip cg">👍 Thumbs Up → Trigger Autocorrect</div>
   </div>
   <div class="msec">☁️ Cloud Architecture</div>
   <div class="chips">
    <div class="chip cb">Browser captures camera frames</div>
    <div class="chip cp">Frames sent via WebSocket to server</div>
    <div class="chip cg">Server runs MediaPipe + TensorFlow</div>
    <div class="chip co">Annotated frames returned to browser</div>
    <div class="chip ck">TTS audio streamed back as base64 MP3</div>
   </div>
  </div>
 </div>
</div>

<script>
// ── State WebSocket ────────────────────────────────────
const SK=['q','w','e','r','t'],AK=['1','2','3','4','5'];
const SC=['s0','s1','s2','s3','s4'],AC=['ac0','ac1','ac2','ac3','ac4'];
const AL=['Title','Sentence','CAPS','lower','Polished'];
let ws,rt;

function connect(){
  const p=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${p}//${location.host}/ws`);
  ws.onopen=()=>{setBadge(true);setS('st-ws',true,'Connected ✓');clearTimeout(rt);};
  ws.onclose=()=>{setBadge(false);setS('st-ws',false,'Reconnecting …');rt=setTimeout(connect,1500);};
  ws.onerror=()=>ws.close();
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.type==='tts_audio'){playBase64Audio(d.data);return;}
    render(d);
  };
}

function setBadge(on){
  const b=document.getElementById('ws-badge');
  b.textContent=on?'● Connected':'○ Reconnecting …';
  b.className='badge '+(on?'on':'off');
  const ws2=document.getElementById('st-ws2');
  if(ws2){
    const col=on?'var(--green)':'var(--orange)';
    ws2.innerHTML=`<span class="sd" style="background:${col}"></span><span style="color:${col};font-family:var(--mono);font-size:.72rem">${on?'Connected ✓':'Connecting …'}</span>`;
  }
}

function render(d){
  const lt=document.getElementById('live');
  lt.innerHTML=d.live_text?x(d.live_text)+'<span class="cur"></span>':'<span class="live-ph">Start signing …</span><span class="cur"></span>';
  const sb2=document.getElementById('sbar'),st=document.getElementById('sbar-txt');
  if(d.status){st.textContent=d.status;sb2.style.display='flex';sb2.className='sbar '+(d.status.includes('✓')?'ok':d.status.includes('⏳')?'warn':'info');}
  else sb2.style.display='none';
  const sc=document.getElementById('sugg-card'),sr=document.getElementById('sr');
  if(d.word_suggestions&&d.word_suggestions.length){
    sc.style.display='block';
    sr.innerHTML=d.word_suggestions.map((w,i)=>`<div class="sc ${SC[i]}" onclick="send('pick_word',${i})"><span class="skey">${SK[i].toUpperCase()}</span>${x(w)}</div>`).join('');
  }else sc.style.display='none';
  const ac=document.getElementById('ac-card'),ab=document.getElementById('ac-body');
  if(d.show_autocorrect&&d.autocorrect_suggestions&&d.autocorrect_suggestions.length){
    ac.style.display='block';
    ab.innerHTML=d.autocorrect_suggestions.map((o,i)=>`<div class="ao" onclick="send('pick_ac',${i})"><div class="an ${AC[i]}">[${i+1}]</div><div class="al">${AL[i]}</div><div class="av">${x(o.length>65?o.slice(0,63)+'…':o)}</div></div>`).join('');
  }else ac.style.display='none';
  const hb=document.getElementById('hb'),hc=document.getElementById('hc');
  if(d.history&&d.history.length){
    hc.textContent=d.history.length+' sentence'+(d.history.length>1?'s':'');
    hb.innerHTML=[...d.history].reverse().slice(0,6).map(s=>`<div class="hi">${x(s)}</div>`).join('');
  }else{hc.textContent='0 sentences';hb.innerHTML='<div class="hi-ph">No sentences yet</div>';}
  setS('st-md',d.model_ready,'MobileNetV2 ✓');
  setS('st-tl',d.tools_ready,'LangTool ✓');
  if(d.translation&&d.translation.text) updateTranslation(d.translation);
}

function setS(id,ok,lbl){
  [id, id+'2'].forEach(eid => {
    const e=document.getElementById(eid);
    if(!e) return;
    const col=ok?'var(--green)':'var(--orange)';
    e.innerHTML=`<span class="sd" style="background:${col}"></span><span style="color:${col};font-family:var(--mono);font-size:.72rem">${ok?lbl:'Loading …'}</span>`;
  });
}

async function send(action,index=0){
  try{await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,index})});}
  catch(e){console.warn(e);}
}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
  if(document.getElementById('modal').classList.contains('open')) return;
  const k=e.key.toLowerCase();
  if(k==='backspace'||k==='z'){e.preventDefault();send('backspace');}
  else if(k==='c') send('clear');
  else if(SK.includes(k)) send('pick_word',SK.indexOf(k));
  else if(AK.includes(k)) send('pick_ac',AK.indexOf(k));
});

function openModal(){document.getElementById('modal').classList.add('open');}
function closeModal(){document.getElementById('modal').classList.remove('open');}
document.getElementById('modal').addEventListener('click',e=>{if(e.target===e.currentTarget)closeModal();});
function x(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// ── Audio playback (gTTS base64 MP3 from server) ──────
function playBase64Audio(b64){
  try{
    const bin=atob(b64),buf=new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) buf[i]=bin.charCodeAt(i);
    const blob=new Blob([buf],{type:'audio/mpeg'});
    const url=URL.createObjectURL(blob);
    const audio=new Audio(url);
    audio.onended=()=>URL.revokeObjectURL(url);
    audio.play().catch(console.warn);
  }catch(e){console.warn('audio:',e);}
}

// ── Browser Camera → /ws/camera ───────────────────────
let camWs,videoEl,canvasEl,ctx2d,capturing=false,waitingForReply=false;

async function startCamera(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720},audio:false});
    videoEl=document.createElement('video');
    videoEl.srcObject=stream;videoEl.autoplay=true;videoEl.playsInline=true;
    videoEl.onloadedmetadata=()=>{
      canvasEl=document.getElementById('cam-canvas');
      canvasEl.width=videoEl.videoWidth;canvasEl.height=videoEl.videoHeight;
      ctx2d=canvasEl.getContext('2d');
      document.getElementById('cam-perm').style.display='none';
      canvasEl.style.display='block';
      setS('st-cam',true,'Camera Active ✓');
      openCamWs();
    };
  }catch(e){
    alert('Camera access denied: '+e.message);
  }
}

function openCamWs(){
  const p=location.protocol==='https:'?'wss:':'ws:';
  camWs=new WebSocket(`${p}//${location.host}/ws/camera`);
  camWs.binaryType='arraybuffer';
  camWs.onopen=()=>{capturing=true;sendFrame();};
  camWs.onclose=()=>{capturing=false;setTimeout(openCamWs,2000);};
  camWs.onerror=()=>camWs.close();
  camWs.onmessage=e=>{
    // Server returned annotated JPEG — draw on canvas, then send next frame
    const blob=new Blob([e.data],{type:'image/jpeg'});
    const url=URL.createObjectURL(blob);
    const img=new Image();
    img.onload=()=>{
      ctx2d.drawImage(img,0,0,canvasEl.width,canvasEl.height);
      URL.revokeObjectURL(url);
      waitingForReply=false;
      sendFrame();
    };
    img.src=url;
  };
}

function sendFrame(){
  if(!capturing||!camWs||camWs.readyState!==WebSocket.OPEN||waitingForReply) return;
  if(!videoEl||videoEl.readyState<2) {setTimeout(sendFrame,50);return;}
  const off=document.createElement('canvas');
  off.width=videoEl.videoWidth;off.height=videoEl.videoHeight;
  off.getContext('2d').drawImage(videoEl,0,0);
  off.toBlob(blob=>{
    if(!blob) return;
    blob.arrayBuffer().then(buf=>{
      if(camWs.readyState===WebSocket.OPEN){
        waitingForReply=true;
        camWs.send(buf);
      }
    });
  },'image/jpeg',0.75);
}

// ── Languages ──────────────────────────────────────────
let _langs=[],_activeLang=null,_transText='';

async function loadLanguages(){
  try{const r=await fetch('/languages');const d=await r.json();_langs=d.languages;buildLangGrid();}
  catch(e){console.warn('langs:',e);}
}

function buildLangGrid(){
  const g=document.getElementById('lang-grid');
  g.innerHTML=_langs.map(l=>`<button class="lang-btn" id="lb-${l.code.replace(/[-]/g,'_')}"
    onclick="selectLang('${l.code}','${l.bcp}','${l.name}')" title="${l.name}">
    <span class="lang-flag">${l.flag}</span><span class="lang-name">${l.name}</span></button>`).join('');
}

function selectLang(code,bcp,name){
  _activeLang={code,bcp,name};
  document.querySelectorAll('.lang-btn').forEach(b=>b.classList.remove('active'));
  const btn=document.getElementById('lb-'+code.replace(/[-]/g,'_'));
  if(btn) btn.classList.add('active');
  document.getElementById('trans-lang-tag').textContent=name;
  const histItems=document.querySelectorAll('#hb .hi');
  const src=histItems.length>0?histItems[0].textContent
    :document.getElementById('live').textContent.replace('▊','').trim();
  if(!src){alert('Sign a sentence first, then choose a language!');return;}
  triggerTranslate(src,code,name);
}

async function triggerTranslate(text,lang_code,lang_name){
  const r=document.getElementById('trans-result'),b=document.getElementById('speak-trans-btn');
  r.className='trans-result empty';r.textContent=`⏳ Translating to ${lang_name} …`;b.disabled=true;
  try{await fetch('/translate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,lang_code})});}
  catch(e){r.textContent='Network error: '+e;}
}

function updateTranslation(tr){
  if(!tr||!tr.text) return;
  _transText=tr.text;
  const r=document.getElementById('trans-result'),b=document.getElementById('speak-trans-btn');
  r.className='trans-result';r.textContent=tr.flag+' '+tr.text;b.disabled=false;
}

async function speakTranslation(){
  if(!_transText||!_activeLang){alert('No translation yet.');return;}
  const btn=document.getElementById('speak-trans-btn');
  btn.innerHTML='🔊 Speaking …';btn.disabled=true;
  try{await fetch('/speak_translation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:_transText,lang_bcp:_activeLang.bcp})});}
  catch(e){console.warn(e);}
  setTimeout(()=>{btn.innerHTML='🔊 Speak Translation';btn.disabled=false;},2500);
}

connect();
loadLanguages();
</script>

<!-- MODEL INFO SECTION -->
<section class="model-info">
  <div class="mi-header">
    <div class="mi-eyebrow">System Information</div>
    <h2 class="section-title">How SignatriX AI Works</h2>
    <p class="section-sub">Powered by MobileNetV2 and MediaPipe Hands for real-time ASL recognition</p>
  </div>

  <div class="mi-grid">
    <!-- Left column: details -->
    <div class="mi-col">
      <div class="info-card">
        <div class="info-card-hdr">
          <div class="info-card-icon" style="background:var(--blue-l)">🧠</div>
          <h3>Model Architecture</h3>
        </div>
        <ul class="info-list">
          <li>Base: MobileNetV2 (Fine-Tuned layers 100–155)</li>
          <li>Input Shape: 128 × 128 × 3</li>
          <li>Output Classes: A–Z + SPACE + DOT + AUTOCORRECT</li>
          <li>Feature Vector: n=1280 (flattened)</li>
          <li>Classifier: Fully Connected → Softmax</li>
        </ul>
      </div>

      <div class="info-card">
        <div class="info-card-hdr">
          <div class="info-card-icon" style="background:var(--green-l)">⚙️</div>
          <h3>Recognition Settings</h3>
        </div>
        <ul class="info-list">
          <li>Confidence Threshold: 0.75</li>
          <li>Smoothing Window: 10 frames</li>
          <li>Letter Hold Duration: 1.0 sec</li>
          <li>Special Gesture Hold: 1.2 sec</li>
          <li>WebSocket Frame Rate: adaptive</li>
        </ul>
      </div>

      <div class="info-card">
        <div class="info-card-hdr">
          <div class="info-card-icon" style="background:var(--purple-l)">📡</div>
          <h3>Live System Status</h3>
        </div>
        <div class="status-rows">
          <div class="status-row"><span>WebSocket</span><span id="st-ws2"><span class="sd" style="background:var(--orange)"></span> Connecting …</span></div>
          <div class="status-row"><span>ML Model</span><span id="st-md2"><span class="sd" style="background:var(--text4)"></span> —</span></div>
          <div class="status-row"><span>Autocorrect</span><span id="st-tl2"><span class="sd" style="background:var(--text4)"></span> —</span></div>
          <div class="status-row"><span>TTS Engine</span><span style="color:var(--green);font-family:var(--mono);font-size:.72rem"><span class="sd" style="background:var(--green)"></span> gTTS Cloud ✓</span></div>
        </div>
      </div>

      <div class="info-card">
        <div class="info-card-hdr">
          <div class="info-card-icon" style="background:var(--orange-l)">🧩</div>
          <h3>Tech Stack</h3>
        </div>
        <div class="chips" style="margin-top:4px">
          <div class="chip cb">Python 3.10</div>
          <div class="chip co">TensorFlow / Keras</div>
          <div class="chip ct">MobileNetV2</div>
          <div class="chip cb">MediaPipe Hands</div>
          <div class="chip cg">OpenCV Headless</div>
          <div class="chip cp">FastAPI + Uvicorn</div>
          <div class="chip cp">WebSocket</div>
          <div class="chip ck">gTTS Cloud TTS</div>
          <div class="chip cg">LanguageTool NLP</div>
          <div class="chip cb">Browser Camera API</div>
        </div>
      </div>
    </div>

    <!-- Right column: images -->
    <div class="mi-col">
      <div class="image-card">
        <div class="image-card-hdr">
          <div class="hdr-dot" style="background:var(--blue)"></div>
          <span>ASL Alphabet Chart</span>
        </div>
        <img src="/static/sign_languages.png" alt="ASL Sign Language Alphabet" onerror="this.parentElement.style.display='none'"/>
      </div>

      <div class="image-card">
        <div class="image-card-hdr">
          <div class="hdr-dot" style="background:var(--purple)"></div>
          <span>MobileNetV2 Architecture</span>
        </div>
        <img src="/static/model_architecture.png" alt="Model Architecture Diagram" onerror="this.parentElement.style.display='none'"/>
      </div>

      <div class="info-card">
        <div class="info-card-hdr">
          <div class="info-card-icon" style="background:#FDF2F8">✋</div>
          <h3>Special Gesture Commands</h3>
        </div>
        <ul class="info-list">
          <li>Flat Hand (1.2s hold) → Insert SPACE</li>
          <li>Pinch gesture → End sentence + Speak</li>
          <li>Thumbs Up → Trigger Autocorrect</li>
          <li>Q–T keys → Select word suggestion</li>
          <li>1–5 keys → Pick autocorrect variant</li>
          <li>Z / Backspace → Delete last letter</li>
          <li>C key → Clear all text</li>
        </ul>
      </div>
    </div>
  </div>
</section>

</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HTML)
