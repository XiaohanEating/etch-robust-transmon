"""Fixed-budget Qwen-VL/text/numerical proposal contracts for pilot experiments."""
import base64
import json
import math
from pathlib import Path
from .aliyun import load_credentials,chat

MODEL='qwen3-vl-plus'
SNAP_COUNT=[0]
SNAP_MAP=[]
EXPERIMENT_VERSION='paired-three-arm-v2-snap'
OPERATIONS=({'kind':'resize','sx':0.90,'sy':1.00},
            {'kind':'resize','sx':1.00,'sy':0.90},
            {'kind':'resize','sx':1.10,'sy':1.00},
            {'kind':'resize','sx':1.00,'sy':1.10},
            {'kind':'round','radius_um':2.0},
            {'kind':'round','radius_um':4.0},
            {'kind':'round','radius_um':8.0},
            {'kind':'resize','sx':1.08,'sy':1.08})


def propose_message(summary, image_path=None):
    text=('在现有形状上提出最多 8 个有区别的单步几何操作。'
          '只能从下面枚举的预注册操作集合中逐字选择参数点，不得使用集合外的任何数值：\n'
          +json.dumps(list(OPERATIONS),ensure_ascii=False,sort_keys=True)+'\n'
          '不得改动结区保护带。只返回 JSON："proposals":["kind":"resize","sx":0.9,"sy":1.0},'
          '{"kind":"round","radius_um":4.0}]}。'
          '每项必须与集合中某一项完全相同，最多 8 项，尽量多样。\n器件信息：'
          +json.dumps(summary,ensure_ascii=False,sort_keys=True))
    if image_path is None:return [{'role':'user','content':text}]
    image=Path(image_path)
    data='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode('ascii')
    return [{'role':'user','content':[{'type':'text','text':text},
             {'type':'image_url','image_url':{'url':data}}]}]
def parse_proposals(content):
    if isinstance(content,str):
        stripped=content.strip()
        if stripped.startswith('```'):
            stripped='\n'.join(line for line in stripped.splitlines() if not line.strip().startswith('```'))
        data=json.loads(stripped)
    else:data=content
    proposals=data.get('proposals')
    if not isinstance(proposals,list) or not proposals or len(proposals)>8:
        raise ValueError('Expected one to eight structured proposals')
    snapped = [0]
    snap_map = []
    for i, op in enumerate(proposals):
        if not isinstance(op,dict) or op.get('kind') not in ('resize','round'):
            raise ValueError('Proposal uses an unsupported operation')
        fields={'kind','sx','sy'} if op['kind']=='resize' else {'kind','radius_um'}
        if set(op)!=fields:raise ValueError('Proposal has missing or extra parameters')
        values=[v for k,v in op.items() if k!='kind']
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            raise ValueError('Proposal parameters must be finite numbers')
        # v2：确定性最近邻吸附到预注册集合（两 VLM 臂同规则；偏离如实计数）
        if not any(op == allowed for allowed in OPERATIONS):
            if op['kind'] == 'resize':
                best = min((a for a in OPERATIONS if a['kind'] == 'resize'),
                           key=lambda a: (a['sx'] - op['sx'])**2 + (a['sy'] - op['sy'])**2)
            else:
                best = min((a for a in OPERATIONS if a['kind'] == 'round'),
                           key=lambda a: abs(a['radius_um'] - op['radius_um']))
            snap_map.append(dict(index=i, raw=op, snapped_to=dict(best)))
            proposals[i] = best
            snapped[0] += 1
    SNAP_COUNT[0] = snapped[0]
    SNAP_MAP.clear()
    SNAP_MAP.extend(snap_map)
    return tuple(proposals)


def call_arm(summary,image_path=None,model=MODEL):
    credentials=load_credentials()
    result=chat(credentials,model,propose_message(summary,image_path),max_tokens=900,timeout=90)
    parsed=parse_proposals(result['content'])
    return dict(model=result['model'],request_id=result['request_id'],usage=result['usage'],
                content=result['content'],snap_map=list(SNAP_MAP),
                n_proposed=len(parsed),proposals=list(parsed))
