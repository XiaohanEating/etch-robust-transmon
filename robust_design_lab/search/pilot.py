"""Equal candidate-budget orchestration; refuses scientific ranking before G1."""
import json
from pathlib import Path
from ..experiments.calibration import representatives
from ..evaluator.participation import VERSION
from ..geometry.operations import apply,VERSION as OP_VERSION
from ..runtime.records import ROOT,provenance,write_json
from ..vlm.experiment import EXPERIMENT_VERSION,OPERATIONS,call_arm
from ..experiments.rendering import render_triptych


def run(output=None,allow_uncalibrated_workflow=False):
    output=Path(output or ROOT/'results'/'pilot_v2');output.mkdir(parents=True,exist_ok=True)
    gate=json.loads((ROOT/'results'/'calibration_v2'/'report.json').read_text(encoding='utf-8'))
    if not gate.get('passed') and not allow_uncalibrated_workflow:
        raise RuntimeError('G1 is not passed: proposal workflow may be tested, but no drift search may run')
    if not gate.get('passed'):
        raise RuntimeError('Workflow dry-run is available; performance evaluation remains disabled until G1')
    record=next(r for r in representatives() if r.category=='rect_calib')
    image=output/'initial.png'
    visual=render_triptych(record,image)
    from ..evaluator import fast,participation
    fs=fast.solve_fields(record)
    metrics=participation.surface_participation(fs,record)
    summary=dict(category=record.category,geometry_hash=record.geometry_hash,
       C_q_fF=metrics['C_q_fF'],f01_GHz=metrics['f01_GHz'],alpha_MHz=metrics['alpha_MHz'],
       T1_us=metrics['T1_us'],field_quantiles=visual['field_log_quantiles'],
       field_change_quantiles=visual['difference_log_quantiles'])
    arms={}
    for name,path in (('no_image',None),('image',image)):
        response=call_arm(summary,path)
        arms[name]=response
    arms['numerical']={'proposals':OPERATIONS}
    records=[]
    for name,arm in arms.items():
        rows=[]
        for index,op in enumerate(arm['proposals']):
            try:
                candidate=apply(record,op)
                rows.append(dict(index=index,operation=op,hash=candidate.geometry_hash,legal=True))
            except Exception as error:
                rows.append(dict(index=index,operation=op,legal=False,reason=str(error)))
        records.append(dict(arm=name,rows=rows,candidate_budget=len(arm['proposals'])))
    report=dict(version=EXPERIMENT_VERSION,model=visual and arms['image']['model'],
       initial_hash=record.geometry_hash,operation_version=OP_VERSION,
       evaluator_version=VERSION,metric_gate=gate,visualization=visual,
       search_enabled=bool(gate.get('passed')),arms=records,provenance=provenance())
    write_json(output/'pilot.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},ensure_ascii=False))


if __name__=='__main__':run()
