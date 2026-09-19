#!/usr/bin/env python3
"""Read-only closeout for the consumed Jia 2025 B-grade author-like test."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage

PAPER = {"Fu": {"R2_log":.69,"MAE_log":.30,"RMSE_log":.41,"GMFE":2.01,"within_2fold":.60,"model_id":"paper_svr_merged"},
         "CL": {"R2_log":.48,"MAE_log":.31,"RMSE_log":.42,"GMFE":2.00,"within_2fold":.64,"model_id":"paper_consensus_15"},
         "VDss":{"R2_log":.60,"MAE_log":.28,"RMSE_log":.35,"GMFE":1.88,"within_2fold":.62,"model_id":"paper_consensus_15"}}
METRICS=["R2_log","MAE_log","RMSE_log","GMFE","within_2fold"]

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--final',type=Path,default=ROOT/'results/final/jia2025_author_like_r3b_one_time_score_v5'); p.add_argument('--r3a',type=Path,default=ROOT/'models/jia2025_author_like_r3a_v1'); p.add_argument('--gate4',type=Path,default=ROOT/'data/public_development/cross_gate_candidate_freeze_v1'); p.add_argument('--output',type=Path,default=ROOT/'results/analysis/jia2025_author_like_r3c_closeout_v2'); a=p.parse_args()
 startup_self_check([a.final/'complete.json',a.r3a/'complete.json',a.gate4/'complete.json'],output=a.output); final=verify_stage(a.final,'jia2025_author_like_r3b_one_time_descriptive_score'); verify_stage(a.r3a,'jia2025_author_like_r3a_traincv_and_model_freeze'); verify_stage(a.gate4,'cross_gate_endpoint_candidate_freeze')
 if not final.get('author_test_scored') or not final.get('rerun_prohibited'): raise ValueError('Final lifecycle is not closed')
 with stage_output(a.output) as out:
  result=pd.read_csv(a.final/'author_test_metrics.csv'); cv=pd.read_csv(a.r3a/'train_cv_metrics.csv'); strict=pd.read_csv(a.gate4/'endpoint_candidate_freeze_registry.csv')
  rows=[]
  for endpoint,spec in PAPER.items():
   x=result[(result.endpoint==endpoint)&(result.model_id==spec['model_id'])&(result.cohort=='author_native_record')].iloc[0]
   for metric in METRICS: rows.append({'endpoint':endpoint,'model_id':spec['model_id'],'metric':metric,'published_jia2025':spec[metric],'author_like_r3b':float(x[metric]),'difference_author_like_minus_published':float(x[metric]-spec[metric]),'comparison_level':'B_author_like_same_workbook_author_membership'})
  comparison=pd.DataFrame(rows); comparison.to_csv(out/'published_vs_author_like_primary.csv',index=False)
  native=[]
  for endpoint,spec in PAPER.items():
   z=result[(result.endpoint==endpoint)&(result.model_id==spec['model_id'])]
   for metric in METRICS:
    n=float(z[z.cohort=='author_native_record'][metric].iloc[0]); q=float(z[z.cohort=='author_parent_purged_sensitivity'][metric].iloc[0]); native.append({'endpoint':endpoint,'model_id':spec['model_id'],'metric':metric,'native':n,'parent_purged':q,'difference_purged_minus_native':q-n})
  pd.DataFrame(native).to_csv(out/'native_vs_parent_purged_sensitivity.csv',index=False)
  strict.loc[strict.endpoint.isin(['fu','CL','VDss']),['endpoint','train_records','train_parents','train_cv_score','primary_metric','comparison_level_current_project','test_lifecycle']].to_csv(out/'project_strict_protocol_context.csv',index=False)
  plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'}); fig,axes=plt.subplots(1,5,figsize=(15,3.4)); labels=['R² (log10)','MAE (log10)','RMSE (log10)','GMFE','Within 2-fold']; endpoints=['Fu','CL','VDss']; xpos=np.arange(3); width=.34
  for ax,metric,label in zip(axes,METRICS,labels):
   d=comparison[comparison.metric==metric].set_index('endpoint').loc[endpoints]; ax.bar(xpos-width/2,d.published_jia2025,width,label='Jia et al. 2025',color='#4C78A8'); ax.bar(xpos+width/2,d.author_like_r3b,width,label='This B-grade author-like run',color='#F58518'); ax.set_xticks(xpos,endpoints); ax.set_title(label); ax.grid(axis='y',alpha=.25); ax.spines[['top','right']].set_visible(False)
  axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.32),ncol=2,frameon=False); fig.text(.5,.01,'Same workbook and author membership; not a strict scaffold/source-cluster comparison.',ha='center'); fig.tight_layout(rect=(0,.06,1,.92)); fig.savefig(out/'Figure_1_Jia2025_published_vs_author_like.png',dpi=600,bbox_inches='tight'); plt.close(fig)
  table=pd.DataFrame([['Jia published /\nauthor-like','Human IV\nFu, CL, VDss','Author split;\nparent/scaffold overlap','B-grade same-workbook\ncontextual comparison'],['Project strict\nprotocol','Human endpoints;\ndistinct project datasets','Predeclared scaffold /\nsource-cluster control','Separate column;\nno cross-protocol rank'] ],columns=['Evidence stream','Scope','Split and leakage status','Permitted claim'])
  fig,ax=plt.subplots(figsize=(13.5,3.4)); ax.axis('off'); t=ax.table(cellText=table.values,colLabels=table.columns,cellLoc='left',colLoc='left',loc='center',colWidths=[.22,.19,.31,.28]); t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1.03,3.0); [cell.set_facecolor('#D9EAF7') for (r,c),cell in t.get_celld().items() if r==0]; ax.set_title('Figure 2. Evidence boundaries for Jia 2025 and project strict protocols',pad=16); fig.savefig(out/'Figure_2_Jia2025_evidence_boundaries.png',dpi=600,bbox_inches='tight'); plt.close(fig)
  report='# Jia 2025 R3c read-only closeout\n\nThe one-time author test is closed. CL consensus nearly matches the published result; Fu and VDss are close. These findings support B-grade author-like reproduction only. Project strict protocol outcomes are intentionally separated and cannot be ranked against this table. No model, calibration, candidate, prediction, or test metric was changed.\n'; (out/'closeout_report.md').write_text(report,encoding='utf-8')
  finish_stage(out,'jia2025_author_like_r3c_read_only_closeout',inputs={str((a.final/'complete.json').resolve()):sha256(a.final/'complete.json'),str((a.r3a/'complete.json').resolve()):sha256(a.r3a/'complete.json'),str((a.gate4/'complete.json').resolve()):sha256(a.gate4/'complete.json')},read_only=True,author_test_rerun=False,model_selection_changed=False,figure_language='English',dpi=600,partial=False)
 print(f'Jia 2025 R3c read-only closeout: {a.output}')
if __name__=='__main__': run_cli(main)
