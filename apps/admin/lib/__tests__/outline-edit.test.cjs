const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),Module=require('node:module'),ts=require('typescript');
const filename=require.resolve('../outline-edit.ts'),mod=new Module(filename,module);
mod._compile(ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,filename);
const {splitRing,splitOutlineTool,placedOutline,verticesInBox}=mod.exports;
const square=[[0,0],[10,0],[10,10],[0,10]];
const area=p=>Math.abs(p.reduce((s,a,i)=>{const b=p[(i+1)%p.length];return s+a[0]*b[1]-a[1]*b[0]},0)/2);
test('split across an outline makes two rings with the same total area',()=>{const p=splitRing(square,[[5,-2],[5,12]]);assert.equal(p.length,2);assert.deepEqual(p.map(area),[50,50]);});
test('split through existing vertices works',()=>{assert.deepEqual(splitRing(square,[[-1,-1],[11,11]]).map(area),[50,50]);});
test('misses, tangents, short lines and multi-crossing cuts preserve the original',()=>{
 for(const line of [[[12,0],[12,10]],[[0,0],[10,0]],[[5,2],[5,8]]])assert.throws(()=>splitRing(square,line));
 const u=[[0,0],[10,0],[10,10],[7,10],[7,3],[3,3],[3,10],[0,10]];assert.throws(()=>splitRing(u,[[-1,5],[11,5]]));
 assert.deepEqual(square,[[0,0],[10,0],[10,10],[0,10]]);
});
test('rotated and translated parts remain at exactly the original displayed position',()=>{
 const t={id:'a',name:'Tool',polygon_mm:square,polygon_px:square.map(p=>p.map(x=>x*2)),rotation_deg:90,offset_mm:{x:20,y:30},depth_mm:7,clearance_mm:1,include:true,shape:{kind:'rect'},notch:{}};
 const parts=splitOutlineTool(t,[[19,35],[31,35]]);assert.equal(parts.length,2);
 for(const p of parts){assert.equal(p.depth_mm,7);assert.equal(p.clearance_mm,1);assert.equal(p.shape,undefined);assert.equal(p.notch,null);assert.equal(p.edited,true);}
 const points=parts.flatMap(placedOutline);assert(Math.min(...points.map(p=>p[0]))>=20-1e-7);assert(Math.max(...points.map(p=>p[0]))<=30+1e-7);assert(Math.min(...points.map(p=>p[1]))>=30-1e-7);assert(Math.max(...points.map(p=>p[1]))<=40+1e-7);
 assert.equal(parts.reduce((s,p)=>s+p.area_mm2,0),100);assert.notEqual(parts[0].id,parts[1].id);
});
test('selection box works in either direction and includes its boundary',()=>{assert.deepEqual(verticesInBox(square,[-1,-1],[10,0]),[0,1]);assert.deepEqual(verticesInBox(square,[10,0],[-1,-1]),[0,1]);assert.deepEqual(verticesInBox(square,[2,2],[8,8]),[]);});
test('editing nodes on a rotated tool does not move unselected vertices',()=>{
 const {editToolOutline}=mod.exports;
 const t={id:'a',polygon_mm:square,polygon_px:[],rotation_deg:37,offset_mm:{x:20,y:30}};
 const before=placedOutline(t), changed=editToolOutline(t,square.map((p,i)=>i===0?[p[0]-2,p[1]-3]:p)),after=placedOutline(changed);
 for(let i=1;i<4;i++){assert(Math.abs(before[i][0]-after[i][0])<1e-7);assert(Math.abs(before[i][1]-after[i][1])<1e-7);}
});
