// Pure client-side calculations only; no browser, network or personal dataset.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {runInNewContext} from 'node:vm';

const template=readFileSync(new URL('../thesaurus/resources/vivino_atlas_template.html',import.meta.url),'utf8');
const start=template.indexOf('function heatColor('),end=template.indexOf('function details(',start);
assert.ok(start>=0 && end>start);
const grapeStart=template.indexOf('function hasGrape('),grapeEnd=template.indexOf('function plannedWine(',grapeStart);
assert.ok(grapeStart>=0 && grapeEnd>grapeStart);
function fixture(){
  const inputs={country:'',grape:''};
  const context={mode:'countries',byId:id=>({value:inputs[id]||''}),data:{analytics:{unique_wines:[
    {id:'a',country:'ES',region:'rioja',grapes:['cab'],tasted:true},
    {id:'b',country:'ES',region:'rioja',grapes:['cab','syrah'],tasted:true},
    {id:'c',country:'FR',region:'bordeaux',grapes:['syrah'],tasted:true},
    {id:'d',country:'FR',region:'bordeaux',grapes:['cab'],tasted:false},
    {id:'e',country:null,region:'unknown',grapes:['cab'],tasted:true}
  ]}}};
  const functions=runInNewContext(template.slice(grapeStart,grapeEnd)+template.slice(start,end)+'\n({heatColor,heatValues})',context);
  return {inputs,context,...functions,heatValues:rows=>new Map(functions.heatValues(rows))};
}
test('countries count only confirmed distinct fixture wines with a known country',()=>{
  const f=fixture(),values=f.heatValues([{id:'ES'},{id:'FR'}]);
  assert.deepEqual([...values],[['ES',2],['FR',1]]);
});
test('grape geography follows actual wines, including blends',()=>{
  const f=fixture();f.context.mode='grapes';
  assert.deepEqual([...f.heatValues([{id:'syrah'}])],[['ES',1],['FR',1]]);
  f.inputs.country='FR';
  assert.deepEqual([...f.heatValues([{id:'syrah'}])],[['FR',1]]);
});
test('grape selector excludes unknown composition and other varieties',()=>{
  const f=fixture();f.inputs.grape='cab';
  assert.deepEqual([...f.heatValues([{id:'ES'},{id:'FR'}])],[['ES',2]]);
});
test('region and wine search subsets change counts without rating fabricated observations',()=>{
  const f=fixture();f.context.mode='regions';
  assert.deepEqual([...f.heatValues([{id:'bordeaux'}])],[['FR',1]]);
  f.context.mode='wines';
  assert.deepEqual([...f.heatValues([{id:'d'}])],[]);
  assert.deepEqual([...f.heatValues([{id:'a'}])],[['ES',1]]);
});
test('fixed heat scale has a neutral zero and increasing intensity',()=>{
  const f=fixture();assert.equal(f.heatColor(0,5),'#e8e1d9');assert.equal(f.heatColor(5,5),'rgb(123,33,62)');
  assert.equal(f.heatColor(1,0),'#e8e1d9');
  const channels=value=>f.heatColor(value,5).match(/\d+/g).map(Number);
  const a=channels(1),b=channels(4);
  assert.ok(a.every((v,i)=>v>b[i]));
  assert.equal(f.heatColor(5,5),f.heatColor(100,5));
});
