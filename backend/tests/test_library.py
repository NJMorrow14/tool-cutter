"""Tool library API (docs/tool-library.md): jobs, multipart tool upload, normalisation, replace, move, delete."""
import io, json, os, shutil, sys, tempfile, time, unittest
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TC_NO_WARM"] = "1"
_TMP = tempfile.mkdtemp(prefix="tc_library_")
os.environ["TC_LIBRARY_DIR"] = _TMP
import app as A


def _jpeg(w=800, h=600):
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.rectangle(img, (200, 150), (600, 450), (0, 140, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


SQUARE = [[50, 70], [110, 70], [110, 130], [50, 130], [50, 70]]      # 60 x 60, offset, closing point repeated


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lib_", dir=_TMP))
        self._old = A.LIBRARY_DIR
        A.LIBRARY_DIR = self.dir
        self.c = A.app.test_client()

    def tearDown(self):
        A.LIBRARY_DIR = self._old
        shutil.rmtree(self.dir, ignore_errors=True)

    def job(self, name="Acme garage"):
        r = self.c.post("/api/library/jobs", json={"name": name})
        self.assertEqual(r.status_code, 201, r.get_json())
        return r.get_json()

    def upload(self, job_id, meta, photo=None, mask=None):
        data = {"tool": json.dumps(meta), "photo": (io.BytesIO(photo or _jpeg()), "photo.jpg")}
        if mask is not None:
            data["mask"] = (io.BytesIO(mask), "mask.png")
        return self.c.post(f"/api/library/jobs/{job_id}/tools", data=data, content_type="multipart/form-data")

    def test_create_upload_normalise_replace(self):
        j = self.job()
        self.assertTrue(len(j["id"]) == 12 and j["tool_count"] == 0 and j["name"] == "Acme garage")
        ok, mask = cv2.imencode(".png", np.zeros((60, 80), np.uint8))
        r = self.upload(j["id"], {"id": "0123456789ab", "name": "Claw hammer", "thickness_mm": 28,
                                  "polygon_mm": SQUARE, "edited": True,
                                  "scale": {"method": "lidar_plane", "mm_per_px": 0.21}}, mask=mask.tobytes())
        self.assertEqual(r.status_code, 201, r.get_json())
        t = r.get_json()
        self.assertEqual(t["id"], "0123456789ab")
        self.assertEqual(t["job_id"], j["id"])
        self.assertEqual(len(t["polygon_mm"]), 4)                    # closing point dropped
        xs = [p[0] for p in t["polygon_mm"]]; ys = [p[1] for p in t["polygon_mm"]]
        self.assertEqual((min(xs), min(ys)), (0, 0))
        self.assertAlmostEqual(t["area_mm2"], 3600.0, places=3)
        self.assertEqual(t["bbox_mm"], [60.0, 60.0])
        self.assertEqual(t["source"], "phone_photo")
        self.assertTrue(t["edited"])
        self.assertEqual(t["thickness_mm"], 28.0)
        self.assertEqual(t["photo_url"], "/api/library/tools/0123456789ab/photo")
        self.assertTrue((self.dir / j["id"] / t["id"] / "mask.png").is_file())
        thumb = self.c.get(t["thumb_url"])
        self.assertEqual(thumb.mimetype, "image/jpeg")
        self.assertEqual(thumb.headers["Cache-Control"], "no-store")
        img = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (240, 320))
        photo = self.c.get(t["photo_url"])
        self.assertEqual(photo.mimetype, "image/jpeg")
        self.assertEqual(cv2.imdecode(np.frombuffer(photo.data, np.uint8), cv2.IMREAD_COLOR).shape[:2], (600, 800))

        # same id again: replaced (200), created kept, still one tool, old mask gone
        r2 = self.upload(j["id"], {"id": "0123456789ab", "name": "Hammer v2", "thickness_mm": None,
                                   "polygon_mm": [[0, 0], [100, 0], [100, 20], [0, 20]]})
        self.assertEqual(r2.status_code, 200, r2.get_json())
        t2 = r2.get_json()
        self.assertEqual(t2["name"], "Hammer v2")
        self.assertIsNone(t2["thickness_mm"])
        self.assertEqual(t2["created"], t["created"])
        self.assertEqual(t2["bbox_mm"], [100.0, 20.0])
        self.assertFalse((self.dir / j["id"] / t["id"] / "mask.png").exists())
        tools = self.c.get(f"/api/library/jobs/{j['id']}/tools").get_json()["tools"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(self.c.get(f"/api/library/jobs/{j['id']}").get_json()["tool_count"], 1)

        # no id: created with a fresh id and a default name
        r3 = self.upload(j["id"], {"polygon_mm": SQUARE})
        self.assertEqual(r3.status_code, 201)
        self.assertEqual(r3.get_json()["name"], "Tool 2")

        # same id in a different job: 409
        other = self.job("Other")
        r4 = self.upload(other["id"], {"id": "0123456789ab", "polygon_mm": SQUARE})
        self.assertEqual(r4.status_code, 409)

    def test_validation(self):
        j = self.job()
        bad = [{"polygon_mm": [[0, 0], [10, 0]]},                         # too few points
               {"polygon_mm": [[0, 0], [3, 0], [3, 3], [0, 3]]},          # 9 mm2 < 25
               {"polygon_mm": [[0, 0], ["a", 0], [1, 1]]},
               {"polygon_mm": [[0, 0], [1e400, 0], [10, 10]]},
               {"polygon_mm": SQUARE, "thickness_mm": 0},
               {"polygon_mm": SQUARE, "thickness_mm": 301},
               {"polygon_mm": SQUARE, "name": "x" * 81},
               {"polygon_mm": SQUARE, "id": "../../etc"}]
        for meta in bad:
            r = self.upload(j["id"], meta)
            self.assertEqual(r.status_code, 400, meta)
            self.assertIn("error", r.get_json())
        r = self.c.post(f"/api/library/jobs/{j['id']}/tools", data={"tool": json.dumps({"polygon_mm": SQUARE})},
                        content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)                              # photo missing
        r = self.upload(j["id"], {"polygon_mm": SQUARE}, photo=b"not an image")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.c.post("/api/library/jobs", json={}).status_code, 400)
        self.assertEqual(self.c.post("/api/library/jobs", json={"name": "y" * 81}).status_code, 400)
        # a bow-tie is repaired by buffer(0) and the largest lobe kept
        r = self.upload(j["id"], {"polygon_mm": [[0, 0], [40, 40], [40, 0], [0, 50]]})
        self.assertEqual(r.status_code, 201, r.get_json())
        self.assertGreater(r.get_json()["area_mm2"], 25)

    def test_list_newest_first_patch_move_delete(self):
        a = self.job("First"); time.sleep(0.01)
        b = self.job("Second")
        jobs = self.c.get("/api/library/jobs").get_json()["jobs"]
        self.assertEqual([x["id"] for x in jobs], [b["id"], a["id"]])
        r = self.c.patch(f"/api/library/jobs/{a['id']}", json={"name": "Renamed"})
        self.assertEqual(r.get_json()["name"], "Renamed")

        t1 = self.upload(a["id"], {"name": "one", "polygon_mm": SQUARE}).get_json(); time.sleep(0.01)
        t2 = self.upload(a["id"], {"name": "two", "polygon_mm": SQUARE}).get_json()
        tools = self.c.get(f"/api/library/jobs/{a['id']}/tools").get_json()["tools"]
        self.assertEqual([t["id"] for t in tools], [t2["id"], t1["id"]])

        r = self.c.patch(f"/api/library/tools/{t1['id']}", json={"name": "Wrench", "thickness_mm": 12.5,
                                                                  "polygon_mm": [[10, 10], [30, 10], [30, 50], [10, 50]]})
        self.assertEqual(r.status_code, 200, r.get_json())
        p = r.get_json()
        self.assertEqual((p["name"], p["thickness_mm"], p["bbox_mm"], p["area_mm2"]), ("Wrench", 12.5, [20.0, 40.0], 800.0))
        self.assertEqual(self.c.patch(f"/api/library/tools/{t1['id']}", json={"thickness_mm": -1}).status_code, 400)

        r = self.c.patch(f"/api/library/tools/{t1['id']}", json={"job_id": b["id"]})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(r.get_json()["job_id"], b["id"])
        self.assertEqual(self.c.get(f"/api/library/tools/{t1['id']}").get_json()["job_id"], b["id"])
        self.assertEqual([t["id"] for t in self.c.get(f"/api/library/jobs/{b['id']}/tools").get_json()["tools"]], [t1["id"]])
        self.assertEqual(self.c.get(f"/api/library/jobs/{a['id']}").get_json()["tool_count"], 1)
        self.assertEqual(self.c.patch(f"/api/library/tools/{t1['id']}", json={"job_id": "ffffffffffff"}).status_code, 404)

        self.assertEqual(self.c.delete(f"/api/library/tools/{t2['id']}").status_code, 204)
        self.assertEqual(self.c.get(f"/api/library/tools/{t2['id']}").status_code, 404)
        self.assertEqual(self.c.get(f"/api/library/tools/{t2['id']}/photo").status_code, 404)
        self.assertEqual(self.c.delete(f"/api/library/jobs/{b['id']}").status_code, 204)
        self.assertEqual(self.c.get(f"/api/library/jobs/{b['id']}").status_code, 404)
        self.assertEqual(self.c.get(f"/api/library/tools/{t1['id']}").status_code, 404)   # removed with its job
        self.assertEqual(self.c.delete(f"/api/library/jobs/{b['id']}").status_code, 404)

    def test_path_traversal_and_bad_ids(self):
        j = self.job()
        for url in ["/api/library/jobs/..%2F..%2Fetc", "/api/library/jobs/ABCDEF012345", "/api/library/tools/../../app.py",
                    "/api/library/tools/..%2Fjob.json/photo", "/api/library/jobs/123/tools"]:
            r = self.c.get(url)
            self.assertIn(r.status_code, (400, 404), url)
        self.assertEqual(self.c.get("/api/library/jobs/ABCDEF012345").status_code, 400)
        self.assertEqual(self.c.get("/api/library/jobs/abcdef012345").status_code, 404)
        self.assertEqual(self.c.delete("/api/library/jobs/..").status_code, 400)
        self.assertTrue((self.dir / j["id"] / "job.json").is_file())


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
