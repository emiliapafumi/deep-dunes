import pyotb

labels_img = "/data/tt.tif"
vec_train = "/data/vec_train.geojson"
vec_valid = "/data/vec_valid.geojson"
vec_test = "/data/vec_test.geojson"

pyotb.PatchesSelection({
    "in": labels_img,
    "grid.step": 32,
    "grid.psize": 64,
    "strategy": "split",
    "strategy.split.trainprop": 0.80,
    "strategy.split.validprop": 0.10,
    "strategy.split.testprop": 0.10,
    "outtrain": vec_train,
    "outvalid": vec_valid,
    "outtest": vec_test
})

import os
os.environ["OTB_TF_NSOURCES"] = "2"

ge_img = ("/data/GE_aoi1.tif")
out_pth = "/data/output/"

for vec in [vec_train, vec_valid, vec_test]:
    app_extract = pyotb.PatchesExtraction({
        "source1.il": ge_img,
        "source1.patchsizex": 64,
        "source1.patchsizey": 64,
        "source1.nodata": 0,
        "source2.il": labels_img,
        "source2.patchsizex": 64,
        "source2.patchsizey": 64,
        "vec": vec,
        "field": "id"
    })
    name = vec.replace("vec_", "").replace(".geojson", "")
    out_dict = {
        "source1.out": name + "_ge_patches_v4.tif",
        "source2.out": name + "_labels_patches_v4.tif",
    }
    pixel_type = {
        "source1.out": "int16",
        "source2.out": "uint8",
    }
    ext_fname = "gdal:co:COMPRESS=DEFLATE"
    app_extract.write(out_dict, pixel_type=pixel_type, ext_fname=ext_fname)


