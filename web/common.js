const TOKEN_KEY="vms_access_token", USER_KEY="vms_user";
function token(){return localStorage.getItem(TOKEN_KEY)||""}function authHeaders(){return {"Authorization":"Bearer "+token()}}
function saveSession(data){localStorage.setItem(TOKEN_KEY,data.access_token);localStorage.setItem(USER_KEY,JSON.stringify(data.user))}
function user(){try{return JSON.parse(localStorage.getItem(USER_KEY))}catch{return null}}
async function requireSession({ready=true}={}){if(!token()){location.replace('/login');return null}const r=await fetch('/api/me',{headers:authHeaders()});if(!r.ok){localStorage.clear();location.replace('/login');return null}const me=await r.json();saveSession({access_token:token(),user:me});if(ready&&me.face_profile_status!=="ready"){location.replace('/masks');return null}return me}
function logout(){localStorage.clear();location.replace('/login')}
