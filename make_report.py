from pathlib import Path
import json
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
root=Path(__file__).parent
out=root/'evidence'
data=json.loads((out/'results_summary.json').read_text())
pdf=root/'Data Science Capstone Project Report.pdf'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Cover',parent=styles['Title'],fontSize=26,leading=32,textColor=colors.HexColor('#102A43'),alignment=TA_CENTER,spaceAfter=20))
styles.add(ParagraphStyle(name='Sub',parent=styles['Normal'],fontSize=12,leading=18,alignment=TA_CENTER))
story=[Spacer(1,1.0*inch),Paragraph('IBM Data Science Capstone',styles['Cover']),Paragraph('SpaceX Falcon 9 First-Stage Landing Prediction',styles['Heading1']),Paragraph('Prepared by Abdul Rehman',styles['Sub']),Spacer(1,.35*inch),Paragraph('Project objective',styles['Heading2']),Paragraph('Analyze launch characteristics and estimate whether the Falcon 9 first stage lands successfully. The workflow includes exploratory data analysis, SQL analysis, launch-site mapping and supervised classification.',styles['BodyText']),Spacer(1,12),Paragraph(f"Dataset: {data['rows']} records and {data['columns']} columns. Observed landing success rate: {data['landing_success_rate_pct']}%.",styles['BodyText']),PageBreak()]
def add_img(name,title):
 p=out/name
 if p.exists():
  story.append(Paragraph(title,styles['Heading2']))
  story.append(Image(str(p),width=7.1*inch,height=3.8*inch))
  story.append(Spacer(1,10))
story += [Paragraph('1. Data and methodology',styles['Heading1']),Paragraph('The course dataset contains launch date, launch site, payload mass, orbit, booster configuration and the binary landing outcome. The target is Class. Outcome is excluded from predictors to avoid target leakage. Data is split into stratified training and test subsets.',styles['BodyText']),Paragraph('2. Exploratory analysis',styles['Heading1'])]
add_img('eda_site_outcome.png','Landing outcome by launch site')
add_img('eda_payload_outcome.png','Payload mass and landing outcome')
add_img('eda_year_trend.png','Launch outcomes over time')
add_img('eda_orbit_outcome.png','Landing outcome by orbit')
story.append(PageBreak())
story += [Paragraph('3. SQL analysis',styles['Heading1']),Paragraph('SQLite queries summarize launch counts by site and orbit, average payload by orbit, site-level landing success rates and yearly launch trends. Result CSV files are included in the evidence folder.',styles['BodyText']),Paragraph('4. Geospatial analysis',styles['Heading1']),Paragraph('The Folium map plots launch-site markers using coordinates in the dataset. Ten-kilometre circles are illustrative proximity buffers only; they are not actual landing tracks or recovery vessel routes. Interactive HTML map is included as evidence.',styles['BodyText']),Paragraph('5. Predictive modelling',styles['Heading1'])]
rows=[['Model','Accuracy','Precision','Recall','F1']]
for x in data['metrics']: rows.append([x['model']]+[f"{x[k]:.3f}" for k in ['accuracy','precision','recall','f1']])
t=Table(rows,repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#D9EAF7')),('GRID',(0,0),(-1,-1),.5,colors.grey),('PADDING',(0,0),(-1,-1),7)]))
story.append(t); story.append(Spacer(1,12))
story.append(Paragraph(f"Best model by F1 score: {data['best_model_by_f1']}.",styles['BodyText']))
add_img('model_comparison.png','Model comparison')
add_img('confusion_matrix.png','Confusion matrix for the selected model')
story += [Paragraph('6. Conclusion and limitations',styles['Heading1']),Paragraph('This project demonstrates an end-to-end classification workflow for Falcon 9 landing outcomes. Model scores are hold-out estimates on this dataset and should not be interpreted as operational guarantees. The Folium proximity circles are illustrative. The interactive Dash application is provided as source code and requires local execution with the dataset.',styles['BodyText']),Paragraph('Repository: https://github.com/abdulrehmangondal206-cmd/IBM-Data-science-capstone-spaceX',styles['BodyText'])]
doc=SimpleDocTemplate(str(pdf),pagesize=landscape(A4),rightMargin=35,leftMargin=35,topMargin=30,bottomMargin=30)
doc.build(story)
print(pdf)
