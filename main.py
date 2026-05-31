import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
import folium

# ─────────────────────────────────────
# 1. 시설 데이터
# ─────────────────────────────────────
gdf = gpd.read_file("wonju_facilities.geojson")

gdf["geometry"] = gdf.geometry.centroid
gdf = gdf.to_crs(epsg=5179)

print("시설 수:", len(gdf))

# ─────────────────────────────────────
# 2. 행정동 경계
# ─────────────────────────────────────
dong_gdf = gpd.read_file(
    "hangjeongdong_강원도.geojson"
)

dong_gdf = dong_gdf[
    dong_gdf["sggnm"] == "원주시"
]

dong_gdf = dong_gdf.to_crs(epsg=5179)

# ─────────────────────────────────────
# 3. 원주시 전체 범위
# ─────────────────────────────────────
x_min, y_min, x_max, y_max = dong_gdf.total_bounds

# ─────────────────────────────────────
# 4. 600m 격자
# ─────────────────────────────────────
GRID_SIZE = 600

x_coords = np.arange(x_min, x_max, GRID_SIZE)
y_coords = np.arange(y_min, y_max, GRID_SIZE)

grid_points = [
    Point(x, y)
    for x in x_coords
    for y in y_coords
]

grid_gdf = gpd.GeoDataFrame(
    geometry=grid_points,
    crs="EPSG:5179"
)

# 원주시 내부만 남김
grid_gdf = gpd.sjoin(
    grid_gdf,
    dong_gdf,
    how="inner",
    predicate="within"
)

grid_gdf["동"] = grid_gdf["adm_nm"]

# ─────────────────────────────────────
# 5. 위경도
# ─────────────────────────────────────
grid_latlon = grid_gdf.to_crs(epsg=4326)

grid_gdf["lat"] = grid_latlon.geometry.y
grid_gdf["lon"] = grid_latlon.geometry.x

# ─────────────────────────────────────
# 좌표 추출 함수
# ─────────────────────────────────────
def get_coords(gdf, col, val):

    if col not in gdf.columns:
        return np.array([])

    subset = gdf[gdf[col] == val]

    if len(subset) == 0:
        return np.array([])

    return np.array([
        [p.x, p.y]
        for p in subset.geometry
    ])

# ─────────────────────────────────────
# 거리 계산
# ─────────────────────────────────────
def min_dist(grid, coords):

    if len(coords) == 0:
        return np.full(
            len(grid),
            999999
        )

    tree = cKDTree(coords)

    dist, _ = tree.query(grid)

    return dist

# ─────────────────────────────────────
# 시설 목록
# ─────────────────────────────────────
facilities = {
    "병원": np.vstack([
        c for c in [
            get_coords(gdf, "amenity", "hospital"),
            get_coords(gdf, "amenity", "clinic")
        ]
        if len(c) > 0
    ]) if any(
        len(get_coords(gdf, "amenity", v)) > 0
        for v in ["hospital", "clinic"]
    ) else np.array([]),

    "약국": get_coords(gdf, "amenity", "pharmacy"),
    "학교": get_coords(gdf, "amenity", "school"),
    "공원": get_coords(gdf, "leisure", "park"),
    "버스정류장": get_coords(gdf, "highway", "bus_stop"),
    "마트": get_coords(gdf, "shop", "supermarket"),
    "편의점": get_coords(gdf, "shop", "convenience"),
    "도서관": get_coords(gdf, "amenity", "library"),
    "체육시설": get_coords(gdf, "leisure", "sports_centre")
}

# ─────────────────────────────────────
# 도보 10분 = 800m
# ─────────────────────────────────────
WALK_DISTANCE = 800

grid_arr = np.array([
    [p.x, p.y]
    for p in grid_gdf.geometry
])

for name, coords in facilities.items():

    dist = min_dist(
        grid_arr,
        coords
    )

    grid_gdf[f"score_{name}"] = (
        dist <= WALK_DISTANCE
    ).astype(int)

# 총점
score_cols = [
    f"score_{name}"
    for name in facilities.keys()
]

grid_gdf["score"] = (
    grid_gdf[score_cols]
    .sum(axis=1)
)

# ─────────────────────────────────────
# 등급
# ─────────────────────────────────────
def score_to_label(score):

    if score >= 7:
        return "상"
    elif score >= 4:
        return "중"
    else:
        return "하"

grid_gdf["label"] = (
    grid_gdf["score"]
    .apply(score_to_label)
)

# ─────────────────────────────────────
# 동별 평균
# ─────────────────────────────────────
mean_score = (
    grid_gdf
    .groupby("동")["score"]
    .mean()
    .sort_values(
        ascending=False
    )
)

# ─────────────────────────────────────
# 지도 생성
# ─────────────────────────────────────
m = folium.Map(
    location=[37.34, 127.96],
    zoom_start=11
)

# 행정동 경계선
dong_border = dong_gdf.to_crs(
    epsg=4326
)

folium.GeoJson(
    dong_border,
    style_function=lambda feature: {
        "fillColor": "transparent",
        "color": "#444",
        "weight": 1
    }
).add_to(m)

# 색상
color_map = {
    "상": "blue",
    "중": "orange",
    "하": "red"
}

# ─────────────────────────────────────
# 우측 상단 평균 점수 박스
# ─────────────────────────────────────
dong_mean_html = ""

for dong, score in mean_score.items():

    dong_mean_html += f"""
    <div style="
    margin-bottom:8px;
    border-bottom:1px solid #eee;
    padding-bottom:6px;
    ">
    <b>{dong.split()[-1]}</b><br>
    평균점수: {score:.2f}
    </div>
    """

info_html = f"""
<div style="
position: fixed;
top: 20px;
right: 20px;
width: 240px;
max-height: 500px;
overflow-y: auto;
background: white;
z-index:9999;
padding:12px;
border-radius:10px;
box-shadow:0 0 8px rgba(0,0,0,0.15);
font-size:13px;
">
<b>생활 인프라 평균 점수</b>
<br><br>
{dong_mean_html}
</div>
"""

m.get_root().html.add_child(
    folium.Element(info_html)
)

# ─────────────────────────────────────
# 점 표시 + 클릭 팝업
# ─────────────────────────────────────
for _, row in grid_gdf.iterrows():

    facility_text = "<br>".join([
        f"{name}: {row[f'score_{name}']}점"
        for name in facilities.keys()
    ])

    popup_html = f"""
    <div style="font-size:13px;">
    <b>{row["동"].split()[-1]}</b><br><br>
    총점: {row["score"]}점<br>
    등급: {row["label"]}<br><br>
    {facility_text}
    </div>
    """

    folium.CircleMarker(
        location=[
            row["lat"],
            row["lon"]
        ],
        radius=5,
        color=color_map[row["label"]],
        fill=True,
        fill_opacity=0.7,
        weight=0,
        popup=folium.Popup(
            popup_html,
            max_width=250
        )
    ).add_to(m)

# 저장
m.save(
    "wonju_accessibility.html"
)

print("완료")
print("wonju_accessibility.html 생성됨")