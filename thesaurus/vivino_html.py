"""Self-contained interactive wine map; untrusted fields are text, never HTML."""
import json


def render(rows, points, quality):
    data = json.dumps({'rows': [dict(row, x=point[0], y=point[1]) for row, point in zip(rows, points, strict=True)],
                       'quality': quality}, ensure_ascii=False, allow_nan=False)
    data = data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return TEMPLATE.replace('__DATA__', data)


TEMPLATE = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>Личная карта Vivino</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#182230;font:16px system-ui,sans-serif}
main{max-width:1400px;margin:auto;padding:24px}h1{margin:0 0 8px}p{line-height:1.5}header p{max-width:1000px;color:#465163}
.filters{display:flex;flex-wrap:wrap;gap:14px;margin:20px 0}label{display:flex;flex-direction:column;gap:6px}
input,select,button{font:inherit;border:1px solid #ccd3dc;border-radius:8px;padding:10px;background:white;color:inherit}
button{cursor:pointer}button:focus,circle:focus{outline:3px solid #3589dc}.layout{display:grid;grid-template-columns:2fr 1fr;gap:18px}
.panel{background:white;border-radius:12px;padding:18px;border:1px solid #e1e5eb}svg{width:100%;min-height:280px;background:#f8fafc}
circle{cursor:pointer;stroke:white;stroke-width:1}#list{display:flex;flex-wrap:wrap;gap:8px;max-height:220px;overflow:auto;margin-top:15px}
#card p{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#596677}.note{font-size:14px;color:#596677}
@media(max-width:800px){main{padding:14px}.layout{grid-template-columns:1fr}.filters label{flex:1;min-width:130px}}
</style></head><body><main><header><h1>Личная карта Vivino</h1>
<p>Близость точек отражает текст отзывов и сведения о стиле, типе, стране и регионе вина.
Это исследовательская карта, а не измерение вкуса или рекомендация. Для записей без отзыва использованы только сведения о вине.</p>
</header><div class="filters">
<label>Поиск<input id="search" type="search" placeholder="Вино, винодельня, отзыв"></label>
<label>Страна<select id="country"><option value="">Все страны</option></select></label>
<label>Тип вина<select id="type"><option value="">Все типы</option></select></label>
<label>Моя оценка от<select id="rating"><option value="0">Любая / нет оценки</option><option>3</option><option>3.5</option><option>4</option><option>4.5</option><option>5</option></select></label>
<label>Отзывы<select id="review"><option value="all">Все записи</option><option value="yes">Только с моим отзывом</option></select></label>
<button id="reset">Сбросить фильтры</button></div>
<p id="summary" role="status"></p><div class="layout"><section class="panel"><svg id="map" viewBox="0 0 1000 640" role="group" aria-label="Карта сходства вин"></svg>
<p class="note">Нажмите точку или название вина. Оси — двумерная проекция текстовых признаков TF-IDF/SVD, самостоятельного вкусового значения у них нет.
Цвет: синяя точка — есть личная оценка, серая — оценка отсутствует. Фильтрация не пересчитывает координаты.</p><div id="list"></div></section>
<aside class="panel" id="card" aria-live="polite"><h2>Выберите вино</h2><p>Здесь появятся оценка, отзыв и сведения из вашего экспорта.</p></aside></div>
<p class="note" id="quality"></p><p class="note">Карта содержит личные отзывы. Данные встроены в этот файл; он ничего не отправляет в сеть. Удаление данных в боте не удаляет уже скачанные копии.</p>
</main><script>
const data=__DATA__, rows=data.rows, byId=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
function add(parent,tag,text,css){const node=document.createElement(tag);node.textContent=text;if(css)node.className=css;parent.appendChild(node);return node;}
for(const key of ['country','type']){for(const value of [...new Set(rows.map(r=>r[key]).filter(Boolean))].sort()){const option=document.createElement('option');option.value=value;option.textContent=value;byId(key).appendChild(option);}}
const xs=rows.map(r=>r.x),ys=rows.map(r=>r.y), minx=Math.min(...xs),miny=Math.min(...ys), dx=Math.max(...xs)-minx||1,dy=Math.max(...ys)-miny||1;
let selected=-1;
function card(index){selected=index;const row=rows[index],root=byId('card');root.replaceChildren();add(root,'h2',row.name);add(root,'p',row.winery+' · '+(row.vintage||'год не указан'));
add(root,'p',[row.country,row.region,row.type,row.style].filter(Boolean).join(' · '));add(root,'p','Моя оценка: '+(row.rating??'не указана')+' / 5; общая: '+(row.average??'не указана')+' / 5');
add(root,'h3','Мой отзыв');add(root,'p',row.review||'Отзыв отсутствует. Координаты рассчитаны по сведениям о вине.');}
function draw(){const root=byId('map'),list=byId('list');root.replaceChildren();list.replaceChildren();const search=byId('search').value.toLocaleLowerCase(),country=byId('country').value,type=byId('type').value,min=Number(byId('rating').value);let count=0;
rows.forEach((row,index)=>{if(country&&row.country!==country||type&&row.type!==type||min&&(row.rating===null||row.rating<min)||byId('review').value==='yes'&&!row.review||search&&![row.name,row.winery,row.review,row.region].join(' ').toLocaleLowerCase().includes(search))return;
count++;const point=document.createElementNS(ns,'circle');point.setAttribute('cx',40+(row.x-minx)/dx*920);point.setAttribute('cy',600-(row.y-miny)/dy*560);point.setAttribute('r',selected===index?'11':'7');point.setAttribute('fill',row.rating===null?'#8a95a6':'#246fa8');point.setAttribute('tabindex','0');point.setAttribute('role','button');point.setAttribute('aria-label',row.name);const title=document.createElementNS(ns,'title');title.textContent=row.name+' · '+(row.rating??'без оценки');point.appendChild(title);
const choose=()=>{card(index);draw();};point.addEventListener('click',choose);point.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});root.appendChild(point);const button=add(list,'button',row.name);button.addEventListener('click',choose);});
byId('summary').textContent='Показано '+count+' из '+rows.length+' записей. '+(count?'':'Измените фильтры.');}
for(const id of ['search','country','type','rating','review'])byId(id).addEventListener('input',draw);
byId('reset').addEventListener('click',()=>{for(const id of ['search','country','type'])byId(id).value='';byId('rating').value='0';byId('review').value='all';draw();});
byId('quality').textContent='С отзывами: '+data.quality.reviewed+'. Пропущено без названия: '+data.quality.missing_names+'. Удалено точных дубликатов: '+data.quality.duplicates+'. Некорректных оценок: '+data.quality.invalid_ratings+'. Отсутствующие оценки не заменены нулями.';draw();
</script></body></html>'''
