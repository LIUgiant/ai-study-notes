"""Run the six original D1 main functions with a local Bailian adapter.

No original course sources or .env values are changed. Real requests require
--run-online. Database collections are created and deleted by the course mains
inside a dedicated work/task2/d1 directory. Reports contain synthetic demo data.
"""
import argparse,contextlib,importlib,io,json,logging,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code'))
os.environ['PYTHON_DOTENV_DISABLED']='1'
os.environ['SEEKDB_MODE']='embedded'
os.environ['LANGCHAIN_TRACING_V2']='false'
os.environ['LANGSMITH_TRACING']='false'
NAMES=['d1_1_base','d1_2_multi_turn','d1_3_streaming','d1_4_tool_use_mock','d1_5_tool_use_seekdb','d1_6_agent']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-online',action='store_true');ap.add_argument('--only',type=int,choices=range(1,7));args=ap.parse_args()
 if not args.run_online: ap.error('Real API calls require --run-online')
 import httpx
 from dotenv import dotenv_values
 from openai import OpenAI
 from langchain.chat_models import init_chat_model
 from langchain.agents import create_agent
 from langchain_core.callbacks import BaseCallbackHandler
 from pyseekdb.client.embedding_function import register_embedding_function
 from seekdb_runtime import create_seekdb_client
 from config import Config
 logging.disable(logging.CRITICAL)
 cfg=dotenv_values(ROOT/'code/.env',interpolate=False);key=cfg.get('DASHSCOPE_API_KEY')
 url=cfg.get('DASHSCOPE_BASE_URL') or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
 if not key or url.rstrip('/')!='https://dashscope.aliyuncs.com/compatible-mode/v1': raise SystemExit('Missing key or unverified endpoint')
 # Match the legacy config names used by the course mains, in memory only.
 Config.SILICONFLOW_API_KEY=key;Config.SILICONFLOW_BASE_URL=url
 Config.DASHSCOPE_API_KEY=key;Config.DASHSCOPE_BASE_URL=url
 out=Path(__file__).parent/'reports';out.mkdir(exist_ok=True)
 selected=[args.only] if args.only else list(range(1,7))
 report={'date':datetime.now(timezone.utc).isoformat(),'model':'qwen-plus','embedding':'text-embedding-v4','source':'original D1 main functions; injected model/client factories','experiments':[]}
 target=out/('d1-results.json' if not args.only else f'd1-{args.only}-results.json')
 def save():target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 calls=0;active=None
 def before_request(request):
  nonlocal calls
  calls+=1
  if calls>50: raise RuntimeError('API call budget exceeded')
  if request.url.host!='dashscope.aliyuncs.com':raise RuntimeError('Unexpected host')
  if active is not None:active['http_requests']+=1
 transport=httpx.Client(trust_env=False,follow_redirects=False,timeout=60,event_hooks={'request':[before_request]})
 api=OpenAI(api_key=key,base_url=url,max_retries=0,http_client=transport)
 class Trace(BaseCallbackHandler):
  raise_error=True
  def on_llm_start(self,serialized,prompts,**kwargs): pass
  def on_llm_new_token(self,token,**kwargs):
   if active is not None:active['stream_chunks']+=1
  def on_llm_end(self,response,**kwargs):
   for batch in response.generations:
    for gen in batch:
     msg=getattr(gen,'message',None)
     if msg is not None:active['model_responses'].append({'content':msg.content,'tool_calls':getattr(msg,'tool_calls',[]),'usage':getattr(msg,'usage_metadata',None),'finish_reason':getattr(msg,'response_metadata',{}).get('finish_reason')})
   save()
  def on_tool_end(self,output,**kwargs):
   active['tool_results'].append(getattr(output,'content',str(output)));save()
 trace=Trace();cache={}
 @register_embedding_function
 class BailianEmbedding:
  dimension=1024
  @staticmethod
  def name():return 'd1_bailian_learning'
  def get_config(self):return {}
  @staticmethod
  def build_from_config(config):return BailianEmbedding()
  def __call__(self,documents):
   if isinstance(documents,str):documents=[documents]
   missing=list(dict.fromkeys(d for d in documents if d not in cache))
   for i in range(0,len(missing),10):
    batch=missing[i:i+10];r=api.embeddings.create(model='text-embedding-v4',input=batch,dimensions=1024,encoding_format='float')
    for row in r.data:cache[batch[row.index]]=row.embedding
    active['embedding_calls'].append({'texts':len(batch),'tokens':r.usage.total_tokens});save()
   return [cache[d] for d in documents]
 def model_factory(_original_name,**_original_config):
  return init_chat_model('qwen-plus',model_provider='openai',api_key=key,base_url=url,http_client=transport,max_retries=0,timeout=60,max_tokens=1000,temperature=0,extra_body={'enable_thinking':False},stream_usage=True,callbacks=[trace])
 class DatabaseAdapter:
  def __init__(self,path):self.db=create_seekdb_client(path)
  def __getattr__(self,name):return getattr(self.db,name)
  def create_collection(self,**kwargs):return self.db.create_collection(**kwargs,embedding_function=BailianEmbedding())
  def close(self):self.db.__exit__(None,None,None)
 def agent_factory(**kwargs):
  return create_agent(**kwargs).with_config({'recursion_limit':16,'callbacks':[trace]})
 try:
  for number in selected:
   active={'number':number,'file':NAMES[number-1]+'.py','status':'running','http_requests':0,'stream_chunks':0,'embedding_calls':[],'model_responses':[],'tool_results':[]};report['experiments'].append(active);save()
   module=importlib.import_module('D1.'+NAMES[number-1]);stream=io.StringIO();start=time.monotonic()
   try:
    kwargs={'model_factory':model_factory}
    if number>=5:
     dbpath=ROOT/'work/task2/d1'/f'experiment-{number}';dbpath.mkdir(parents=True,exist_ok=True)
     kwargs.update(client_factory=lambda path:DatabaseAdapter(path),db_path=dbpath)
    if number==6:kwargs['agent_factory']=agent_factory
    with contextlib.redirect_stdout(stream):code=module.main(**kwargs)
    active['status']='passed' if code==0 else 'failed';active['exit_code']=code
   except Exception as exc:
    active['status']='failed';active['error_type']=type(exc).__name__;active['http_status']=getattr(exc,'status_code',None)
   active['seconds']=round(time.monotonic()-start,3);active['stdout']=stream.getvalue();save()
   print(f'D1-{number}: {active["status"]}, {active["http_requests"]} requests',flush=True)
 finally:api.close()
 return 0 if all(x['status']=='passed' for x in report['experiments']) else 1
if __name__=='__main__':raise SystemExit(main())
