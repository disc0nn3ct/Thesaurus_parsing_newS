# Справочники географического атласа Vivino

Снимок от 2026-10-05: 777 записей регионов, 1637 сортов с идентификатором VIVC, 177 записей стран и территорий. Это вспомогательный неполный справочник, а не исчерпывающий реестр виноделия мира. Классификация Wikidata может включать исторические регионы и сорта разного назначения. Отсутствие отметки означает отсутствие подтверждения в личных данных, а не утверждение, что пользователь никогда этого не пробовал.

## Источники и лицензии

- `vivino_catalog.json`: [Wikidata](https://www.wikidata.org/wiki/Wikidata:Licensing), CC0. Регионы отобраны по типу [wine-producing region, Q2140699](https://www.wikidata.org/wiki/Q2140699), включая подклассы; сорта — по наличию [идентификатора VIVC, P3904](https://www.wikidata.org/wiki/Property:P3904). Имена, русские названия, описания, синонимы и координаты берутся из этих записей. Ссылки VIVC открываются по запросу пользователя; весь каталог VIVC не скопирован. Идентификатор не гарантирует, что сорт используется именно для вина.
- `vivino_world.json` и поля стран: [Natural Earth, 1:110m Admin 0 countries](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson), [public domain](https://www.naturalearthdata.com/about/terms-of-use/). Геометрия преобразована в локальные SVG-пути в равнопромежуточной цилиндрической проекции; антимередианные рёбра разделены. Границы служат фоном, не подтверждают принадлежность винодельческих регионов.

Значения `null` координат сохраняются. Точки регионов показывают справочную позицию региона и включаются отдельно; при её отсутствии регион остаётся в списке без точки. Страны по умолчанию отображаются заливкой, соответствующей числу разных подтверждённых вин пользователя. В `vivino_world.json` поле `features` связывает каждый SVG-путь с кодом страны из того же снимка; `paths` сохранено для совместимости. Положение винодельни, места сканирования и пользователя не вычисляется. Регион сопоставляется по известному имени/синониму и стране; неоднозначные совпадения не выбираются автоматически. Сорт сопоставляется только по уникальному известному имени/синониму. Личные дополнения и неизвестные регионы сохраняются отдельно от общего справочника.

## Воспроизведение снимка

Построение приложения не требует загрузки справочников: готовые JSON включены в проект. Ручной запуск `python scripts/build_vivino_catalog.py PATH` читает четыре JSON из `PATH` и обновляет только общие ресурсы. Персональные экспорты в этом процессе не используются. Обновление ресурсов само по себе не импортирует пользовательские файлы и не рассылает карты.

`world.json` — файл из [официального репозитория Natural Earth](https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson). Остальные файлы — JSON-ответы `https://query.wikidata.org/sparql`, параметр `format=json`, запросы ниже. Порядок полей и источники первичных записей сохраняются; несколько строк одного QID объединяются. Если у региона несколько координат, выбирается первая полученная. Новый ответ Wikidata может отличаться от текущего снимка.

`regions.json`:

```sparql
SELECT DISTINCT ?item ?name ?description ?coord ?countryName ?code WHERE {
 ?item wdt:P31/wdt:P279* wd:Q2140699 .
 ?item rdfs:label ?name . FILTER(LANG(?name)="en")
 OPTIONAL { ?item schema:description ?description . FILTER(LANG(?description)="en") }
 OPTIONAL { ?item wdt:P625 ?coord }
 OPTIONAL { ?item wdt:P17 ?country . ?country rdfs:label ?countryName . FILTER(LANG(?countryName)="en") OPTIONAL { ?country wdt:P297 ?code } }
} LIMIT 10000
```

`aliases.json`:

```sparql
SELECT DISTINCT ?item ?ru ?alias WHERE {
 ?item wdt:P31/wdt:P279* wd:Q2140699 .
 OPTIONAL { ?item rdfs:label ?ru . FILTER(LANG(?ru)="ru") }
 OPTIONAL { ?item skos:altLabel ?alias . FILTER(LANG(?alias) IN ("en","ru")) }
} LIMIT 20000
```

`grapes.json`:

```sparql
SELECT DISTINCT ?item ?name ?ru ?description ?vivc ?originName ?alias WHERE {
 ?item wdt:P3904 ?vivc . ?item rdfs:label ?name . FILTER(LANG(?name)="en")
 OPTIONAL { ?item rdfs:label ?ru . FILTER(LANG(?ru)="ru") }
 OPTIONAL { ?item schema:description ?description . FILTER(LANG(?description)="en") }
 OPTIONAL { ?item wdt:P495 ?origin . ?origin rdfs:label ?originName . FILTER(LANG(?originName)="en") }
 OPTIONAL { ?item skos:altLabel ?alias . FILTER(LANG(?alias) IN ("en","ru")) }
} LIMIT 20000
```

Исходные ответы не включаются в Git. При обновлении следует проверить ограничения числа результатов, совпадения названий, идентификаторы, координаты и изменения классификации; запустить `tests/test_vivino_atlas.py`. Не следует представлять этот снимок как полный каталог VIVC или все регионы мира.
