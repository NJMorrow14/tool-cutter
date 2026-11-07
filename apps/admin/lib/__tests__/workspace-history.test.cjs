const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const Module = require('node:module');
const ts = require('typescript');
const filename = require.resolve('../workspace-history.ts');
const compiled = new Module(filename, module);
compiled._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText, filename);
const { createWorkspaceHistory } = compiled.exports;
const shape = () => ({ id: 'a', name: 'Socket', source: 'shape', polygon_mm: [[0,0],[10,0],[10,10]], polygon_px: [], shape: {kind:'rect',w_mm:10,h_mm:10,r_mm:0}, rotation_deg: 0, offset_mm: {x:10,y:20}, include:true, pending:false });
const initial = () => ({ tools: [], settings: { default_clearance_mm: 1 }, matSize: { width_mm:300,height_mm:200 } });
const doc = h => h.getSnapshot().document;
const action = (h, fn) => { h.beginPointer(); fn(); h.endPointer(); };

test('add, move, rename and remove undo chronologically and redo completely', () => {
 const h=createWorkspaceHistory(initial());
 action(h,()=>h.update('tools',[shape()]));
 action(h,()=>h.update('tools',t=>t.map(x=>({...x,offset_mm:{x:40,y:50}}))));
 action(h,()=>h.update('tools',t=>t.map(x=>({...x,name:'Driver'}))));
 action(h,()=>h.update('tools',[]));
 h.undo();assert.equal(doc(h).tools[0].name,'Driver');
 h.undo();assert.equal(doc(h).tools[0].name,'Socket');
 h.undo();assert.deepEqual(doc(h).tools[0].offset_mm,{x:10,y:20});
 h.undo();assert.deepEqual(doc(h).tools,[]);assert.equal(h.getSnapshot().canUndo,false);
 for(let i=0;i<4;i++)h.redo();assert.deepEqual(doc(h).tools,[]);assert.equal(h.getSnapshot().canRedo,false);
});
test('a drag with many frames is one undo and restores parametric shape metadata',()=>{
 const start={...initial(),tools:[shape()]};const h=createWorkspaceHistory(start);
 h.beginPointer();
 for(let i=1;i<=30;i++)h.update('tools',t=>t.map(x=>({...x,polygon_mm:[[i,0],[10,0],[10,10]],shape:undefined,offset_mm:{x:0,y:0},edited:true})));
 h.endPointer();const edited=doc(h);h.undo();assert.deepEqual(doc(h),start);assert.equal(h.getSnapshot().canUndo,false);
 h.redo();assert.deepEqual(doc(h),edited);
});
test('whole field edit and held arrow key are single undo steps',()=>{
 const h=createWorkspaceHistory({...initial(),tools:[shape()]});
 h.beginField();for(const name of ['D','Dr','Driver'])h.update('tools',t=>t.map(x=>({...x,name})));h.endField();
 h.beginKey();for(let i=0;i<5;i++)h.update('tools',t=>t.map(x=>({...x,rotation_deg:x.rotation_deg+90})));h.endKey();
 h.undo();assert.equal(doc(h).tools[0].rotation_deg,0);assert.equal(doc(h).tools[0].name,'Driver');
 h.undo();assert.equal(doc(h).tools[0].name,'Socket');
});
test('settings and dimensions participate in the same history as tools',()=>{
 const h=createWorkspaceHistory(initial());action(h,()=>h.update('tools',[shape()]));
 action(h,()=>h.update('settings',{default_clearance_mm:3}));action(h,()=>h.update('matSize',{width_mm:400,height_mm:250}));
 h.undo();assert.equal(doc(h).matSize.width_mm,300);h.undo();assert.equal(doc(h).settings.default_clearance_mm,1);h.undo();assert.equal(doc(h).tools.length,0);
});
test('no-op updates do not consume history or destroy redo',()=>{
 const h=createWorkspaceHistory(initial());action(h,()=>h.update('tools',[shape()]));h.undo();
 h.update('tools',t=>[...t]);assert.equal(h.getSnapshot().canRedo,true);h.redo();assert.equal(doc(h).tools.length,1);
});
test('editing after undo clears the old redo branch',()=>{
 const h=createWorkspaceHistory(initial());action(h,()=>h.update('tools',[shape()]));h.undo();
 action(h,()=>h.update('settings',{default_clearance_mm:5}));assert.equal(h.getSnapshot().canRedo,false);
});
test('delayed detection result joins its initiating edit',async()=>{
 const h=createWorkspaceHistory(initial());h.beginPointer();h.update('tools',[{...shape(),pending:true,polygon_mm:[]}]);const finish=h.captureTools();h.endPointer();
 await Promise.resolve();finish(t=>t.map(x=>({...x,pending:false,polygon_mm:shape().polygon_mm})));
 h.undo();assert.equal(doc(h).tools.length,0);assert.equal(h.getSnapshot().canUndo,false);
 h.redo();assert.equal(doc(h).tools[0].polygon_mm.length,3);assert.equal(doc(h).tools[0].pending,false);
});
test('undo invalidates late responses and stale component writers, even after redo',()=>{
 const h=createWorkspaceHistory(initial());action(h,()=>h.update('tools',[{...shape(),pending:true}]));
 const finish=h.captureTools(), oldWriter=h.writer('tools');h.undo();h.redo();
 finish([]);oldWriter([]);assert.equal(doc(h).tools.length,1);assert.equal(doc(h).tools[0].pending,false);
});
test('pending/error flags alone are not undoable edits',()=>{
 const h=createWorkspaceHistory({...initial(),tools:[shape()]});h.update('tools',t=>t.map(x=>({...x,pending:true})));
 h.update('tools',t=>t.map(x=>({...x,pending:false,error:'Network unavailable'})));assert.equal(h.getSnapshot().canUndo,false);
});
test('new scan resets history and invalidates prior async writes',()=>{
 const h=createWorkspaceHistory(initial());action(h,()=>h.update('tools',[shape()]));const finish=h.captureTools();
 const next={...initial(),matSize:{width_mm:100,height_mm:100}};h.reset(next);finish([shape()]);assert.deepEqual(doc(h),next);assert.equal(h.getSnapshot().canUndo,false);
});
test('history is bounded',()=>{
 const h=createWorkspaceHistory(initial(),3);for(let i=1;i<=5;i++)action(h,()=>h.update('matSize',{width_mm:i,height_mm:200}));
 for(let i=0;i<4;i++)h.undo();assert.equal(doc(h).matSize.width_mm,2);
});

test('canvas drag stays separate when a property input has not blurred',()=>{
 const h=createWorkspaceHistory({...initial(),tools:[shape()]});h.beginField();
 h.update('tools',t=>t.map(x=>({...x,name:'Driver'})));
 action(h,()=>h.update('tools',t=>t.map(x=>({...x,offset_mm:{x:40,y:50}}))));
 h.undo();assert.equal(doc(h).tools[0].name,'Driver');assert.deepEqual(doc(h).tools[0].offset_mm,{x:10,y:20});
 h.undo();assert.equal(doc(h).tools[0].name,'Socket');
});
