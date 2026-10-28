"""第12讲 复购与固定观察窗客户价值
业务问题：比较客户价值时，大家有同样长的观察时间吗？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

# %% 全期间复购率只是描述，较晚进入的客户没有同样长的观察机会
o=positive_orders();first=o.groupby('customer_id').order_date.min().rename('first_purchase')
o=o.merge(first,on='customer_id',validate='many_to_one')
counts=o.groupby('customer_id').order_id.nunique()
print(f'全期描述性复购比例：{counts.ge(2).mean():.2%}')
results=[];customer_windows=[]
for horizon in [30,60,90]:
    eligible=first.index[first+pd.Timedelta(days=horizon)<=AS_OF]
    x=o.loc[o.customer_id.isin(eligible)&o.order_date.lt(o.first_purchase+pd.Timedelta(days=horizon))]
    by=x.groupby('customer_id').agg(orders=('order_id','nunique'),net_sales_cents=('net_sales_calc','sum'),contribution_cents=('contribution_cents','sum'))
    by=by.reindex(eligible,fill_value=0);by['horizon_days']=horizon
    results.append({'horizon_days':horizon,'eligible_customers':len(by),'repeat_rate':by.orders.ge(2).mean(),
        'mean_observed_value_yuan':by.net_sales_cents.mean()/100,'mean_observed_contribution_yuan':by.contribution_cents.mean()/100})
    customer_windows.append(by.reset_index())
report=pd.DataFrame(results);show(report)
save_table(report,12,'mature_window_summary');save_table(pd.concat(customer_windows),12,'customer_windows')
fig,ax=plt.subplots(figsize=(7,4));ax.plot(report.horizon_days,report.repeat_rate*100,marker='o',color=BLUE)
ax.set(title='Repeat purchase among fully observed customers',xlabel='Observation window (days)',ylabel='Repeat purchase (%)');save_figure(fig,12,'repeat_windows')
print('这里是30/60/90天已观测价值，不是预测终身LTV；不同窗口的合格客户群也不同。')
