"""Execute the workflow selectors against colliding fork/main artifact names."""
import json
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('workflow,step_id,expected',[
    ('event_research.yml','previous',{'run':4}),
    ('event_hourly.yml','state',{'research':4,'hourly':4}),
])
def test_model_restoration_rejects_foreign_repository_artifacts(workflow,step_id,expected):
    root=Path(__file__).resolve().parents[1]
    document=(root/'.github/workflows'/workflow).read_text()
    block=document.split(f'id: {step_id}\n',1)[1].split('          script: |\n',1)[1]
    lines=[]
    for line in block.splitlines():
        if not line.startswith('            '):break
        lines.append(line[12:])
    script='\n'.join(lines)
    # A fork can name its branch main. Missing provenance must also fail closed.
    candidates=[
        {'id':1,'head_repository_id':999,'head_branch':'main'},
        {'id':2,'head_branch':'main'},
        {'id':4,'head_repository_id':123,'head_branch':'main'},
    ]
    # The publisher must also reject a same-repository PR run called main.
    if step_id=='state':
        candidates.insert(2,{'id':3,'head_repository_id':123,'head_branch':'main'})
    harness=r'''
const input=JSON.parse(process.argv[1]),outputs={},lookups=[];
const context={eventName:'push',sha:'merged',repo:{owner:'owner',repo:'repo'},payload:{repository:{id:123}}};
const core={setOutput:(key,value)=>{outputs[key]=value;}};
const github={rest:{repos:{listPullRequestsAssociatedWithCommit:async()=>({data:[]})},actions:{
  listArtifactsForRepo:async()=>({data:{artifacts:input.candidates.map(workflow_run=>({expired:false,workflow_run}))}}),
  getWorkflowRun:async({run_id})=>{
    lookups.push(run_id);
    return {data:{path:'.github/workflows/'+(outputs.research?'event_hourly.yml':'event_research.yml'),
      event:run_id===3?'pull_request':'push',conclusion:'success'}};
  }
}}};
(async()=>{
  await new (Object.getPrototypeOf(async function(){}).constructor)('github','context','core',input.script)(github,context,core);
  console.log(JSON.stringify({outputs,lookups}));
})().catch(error=>{console.error(error);process.exit(1);});
'''
    result=subprocess.run(['node','-e',harness,json.dumps({'script':script,'candidates':candidates})],
                          check=True,capture_output=True,text=True)
    actual=json.loads(result.stdout)
    for key,value in expected.items():
        assert actual['outputs'][key]==value
    assert not ({1,2}&set(actual['lookups']))
