const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),Module=require('node:module'),ts=require('typescript');
const filename=require.resolve('../viewport.ts'),mod=new Module(filename,module);
mod._compile(ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,filename);
const {zoomViewport,frameViewport}=mod.exports;
const fit={x:-6,y:-6,w:292,h:228};
test('zoom keeps the cursor anchor at the same position on screen',()=>{
 const anchor={x:43,y:81},next=zoomViewport(fit,anchor,2,fit.w);
 assert.equal(next.w,146);assert.equal(next.h,114);
 assert.equal((anchor.x-fit.x)/fit.w,(anchor.x-next.x)/next.w);
 assert.equal((anchor.y-fit.y)/fit.h,(anchor.y-next.y)/next.h);
 assert.deepEqual(zoomViewport(next,anchor,.5,fit.w),fit);
});
test('zoom clamps at usable minimum and maximum magnifications',()=>{
 assert.equal(zoomViewport(fit,{x:0,y:0},1e10,fit.w).w,fit.w/32);
 assert.equal(zoomViewport(fit,{x:0,y:0},1e-10,fit.w).w,fit.w/.25);
});
test('frame selected includes all vertices and preserves the viewport aspect ratio',()=>{
 const p=[[10,20],[30,20],[30,90],[10,90]],v=frameViewport(p,fit);
 assert(Math.abs(v.w/v.h-fit.w/fit.h)<1e-10);
 for(const [x,y]of p)assert(x>=v.x&&x<=v.x+v.w&&y>=v.y&&y<=v.y+v.h);
 assert.deepEqual(frameViewport([],fit),fit);
});
