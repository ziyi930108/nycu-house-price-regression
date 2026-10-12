"""Create a connected flowchart as PNG and editable SVG; no model execution."""
from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / 'assets'
OUT.mkdir(exist_ok=True)
W, H = 1800, 1180
image = Image.new('RGB', (W, H), 'white')
draw = ImageDraw.Draw(image)
font_path = 'C:/Windows/Fonts/msjh.ttc'
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#4A5665"/></marker></defs>',
       '<rect width="100%" height="100%" fill="white"/>']

def font(size, bold=False):
    return ImageFont.truetype(font_path, size=size, index=1 if bold else 0)

def box(x, y, w, h, lines, fill='#EAF0F6', size=35):
    draw.rounded_rectangle((x,y,x+w,y+h), radius=14, fill=fill, outline='#8796A7',width=3)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#8796A7" stroke-width="3"/>')
    gap=size*1.33; top=y+h/2-gap*(len(lines)-1)/2
    for i,line in enumerate(lines):
        yy=top+i*gap
        draw.text((x+w/2,yy),line,font=font(size,i==0),fill='#142C44',anchor='mm')
        svg.append(f'<text x="{x+w/2}" y="{yy}" font-family="Microsoft JhengHei" font-size="{size}" text-anchor="middle" dominant-baseline="central" font-weight="{"bold" if i==0 else "normal"}" fill="#142C44">{escape(line)}</text>')

def arrow(points):
    draw.line(points,fill='#4A5665',width=5,joint='curve')
    a,b=points[-2],points[-1]
    import math
    ang=math.atan2(b[1]-a[1],b[0]-a[0]); k=19
    tri=[b,(b[0]-k*math.cos(ang-.46),b[1]-k*math.sin(ang-.46)),(b[0]-k*math.cos(ang+.46),b[1]-k*math.sin(ang+.46))]
    draw.polygon(tri,fill='#4A5665')
    svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#4A5665" stroke-width="5" marker-end="url(#arrow)"/>')

# All connectors are drawn before nodes, so the node frames stay clear.
arrow([(310,110),(310,150),(900,150),(900,180)])
arrow([(900,110),(900,180)])
arrow([(1490,110),(1490,150),(900,150),(900,180)])
arrow([(900,280),(900,325)])
arrow([(900,425),(900,465),(310,465),(310,505)])
arrow([(900,425),(900,505)])
arrow([(900,425),(900,465),(1490,465),(1490,505)])
for x in [310,900,1490]: arrow([(x,625),(x,670)])
arrow([(310,770),(310,815),(900,815),(900,860)])
arrow([(900,770),(900,860)])
arrow([(1490,770),(1490,815),(900,815),(900,860)])
arrow([(900,960),(900,1010)])
box(50,10,520,100,['train：房屋條件＋總價'])
box(640,10,520,100,['valid：最終合併標籤'])
box(1230,10,520,100,['test：房屋條件，無總價'])
box(200,180,1400,100,['表頭檢查與文字解析；僅訓練資料完全去重','test 的每筆編號與原順序完整保留'],size=33)
box(200,325,1400,100,['建立特徵及 fit 狀態 → 套用到 query','詞表、數值轉換、價格參考與道路資訊'],size=33)
box(50,505,520,120,['單價模型','CatBoost ＋ LightGBM'])
box(640,505,520,120,['神經網路','面積網路 ＋ 參考單價網路'])
box(1230,505,520,120,['總價模型','CatBoost總價 ＋ 對數總價'])
box(50,670,520,100,['先還原總價 → 單價路線','75% C_unit ＋ 25% L_unit'],size=29)
box(640,670,520,100,['先還原總價 → 網路路線','75%雙NN均值 ＋ 25% L_unit'],size=28)
box(1230,670,520,100,['先還原總價 → 總價路線','75%雙總價均值 ＋ 25% L_unit'],size=28)
box(200,860,1400,100,['各路分別套用成交參考 R 與空間修正 S','S 同時接收原始價格 B 與 R(B)'],size=33)
box(200,1010,1400,125,['三路逐筆取中位數 → 一次取整 → CSV檢查與輸出','驗證流程讀 query 真價算 MAE；test 真價未知'],fill='#E8F2EE',size=33)
svg.append('</svg>')
image.save(OUT/'program_flow.png')
(OUT/'program_flow.svg').write_text('\n'.join(svg),encoding='utf8')
print(OUT/'program_flow.png')
