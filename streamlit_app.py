# streamlit_app.py
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
import streamlit as st

# 🔵 Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 🔤 한글 폰트 (Pretendard-Bold.ttf)
from matplotlib import font_manager as fm, rcParams
from pathlib import Path
font_path = Path("fonts/Pretendard-Bold.ttf").resolve()
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    rcParams["font.family"] = font_prop.get_name()
else:
    font_prop = fm.FontProperties()
rcParams["axes.unicode_minus"] = False

# -------------------------------------------------
# ✅ ERDDAP: SOEST Hawaii 인스턴스 한 곳만 사용 (고정)
#   - OISST v2.1 (AVHRR) anomaly 포함
#   - 이 인스턴스는 현재 2024-12-31까지 제공됨
# -------------------------------------------------
ERDDAP_URL = "https://erddap.aoml.noaa.gov/hdb/erddap/griddap/SST_OI_DAILY_1981_PRESENT_T"

def _open_ds(url_base: str):
    """서버 설정에 따라 .nc 필요할 수 있어 두 번 시도 (동일 엔드포인트 고정)."""
    try:
        return xr.open_dataset(url_base, decode_times=True)
    except Exception:
        return xr.open_dataset(url_base + ".nc", decode_times=True)

def _standardize_anom_field(ds: xr.Dataset, target_time: pd.Timestamp) -> xr.DataArray:
    """
    - 변수: 'anom'
    - 깊이 차원(있다면): 표층 선택
    - 좌표명: latitude/longitude → lat/lon 통일
    - 시간: 데이터 커버리지 바깥이면 경계로 클램프 후 'nearest'
    """
    da = ds["anom"]

    # 깊이 차원 표층 선택
    for d in ["zlev", "depth", "lev"]:
        if d in da.dims:
            da = da.sel({d: da[d].values[0]})
            break

    # 시간 클램프 + nearest (멀리 점프 방지)
    times = pd.to_datetime(ds["time"].values)
    tmin, tmax = times.min(), times.max()
    if target_time < tmin:
        target_time = tmin
    elif target_time > tmax:
        target_time = tmax
    da = da.sel(time=target_time, method="nearest").squeeze(drop=True)

    # 좌표명 통일
    rename_map = {}
    if "latitude" in da.coords:  rename_map["latitude"]  = "lat"
    if "longitude" in da.coords: rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)

    return da

# -----------------------------
# 데이터 접근 (SOEST만 사용)
# -----------------------------
@st.cache_data(show_spinner=False)
def list_available_times() -> pd.DatetimeIndex:
    ds = _open_ds(ERDDAP_URL)
    times = pd.to_datetime(ds["time"].values)
    ds.close()
    return pd.DatetimeIndex(times)

@st.cache_data(show_spinner=True)
def load_anomaly(date: pd.Timestamp, bbox=None) -> xr.DataArray:
    """
    선택 날짜의 anomaly(°C) 2D 필드 반환.
    bbox=(lat_min, lat_max, lon_min, lon_max); 경도 -180~180.
    날짜 변경선 횡단 시 자동 분할-결합.
    """
    ds = _open_ds(ERDDAP_URL)
    da = _standardize_anom_field(ds, date)

    # bbox 슬라이스
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox

        # 위도
        if lat_min <= lat_max:
            da = da.sel(lat=slice(lat_min, lat_max))
        else:
            da = da.sel(lat=slice(lat_max, lat_min))

        # 경도 (+ 날짜변경선 처리)
        if lon_min <= lon_max:
            da = da.sel(lon=slice(lon_min, lon_max))
        else:
            left  = da.sel(lon=slice(lon_min, 180))
            right = da.sel(lon=slice(-180, lon_max))
            da = xr.concat([left, right], dim="lon")

    ds.close()
    return da

# -----------------------------
# Cartopy Plot
# -----------------------------
def plot_cartopy_anomaly(
    da: xr.DataArray,
    title: str,
    vabs: float = 5.0,
    projection=ccrs.Robinson(),
    extent=None,
):
    fig = plt.figure(figsize=(12.5, 6.5))
    ax = plt.axes(projection=projection)

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=3)

    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    cmap = cm.get_cmap("RdBu_r").copy()
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    if "lon" in da.coords:
        da = da.sortby("lon")

    im = ax.pcolormesh(
        da["lon"], da["lat"], da.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, shading="auto", zorder=2
    )

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.04, shrink=0.9)
    cbar.set_label("해수면 온도 편차 (°C, 1971–2000 기준)", fontproperties=font_prop)

    ax.set_title(title, pad=8, fontproperties=font_prop)
    fig.tight_layout()
    return fig

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("🛠️ 보기 옵션")

# 날짜 범위 = SOEST 실제 커버리지로 제한
with st.spinner("사용 가능한 날짜 불러오는 중..."):
    times = list_available_times()
tmin, tmax = times.min().date(), times.max().date()

# ✅ 기본 시작일 = 2024-08-15 (커버리지 범위 바깥이면 자동 조정)
DEFAULT_START = pd.Timestamp("2024-08-15")
if DEFAULT_START.date() < tmin:
    default_date = times[0]
elif DEFAULT_START.date() > tmax:
    default_date = times[-1]
else:
    default_date = DEFAULT_START

date = st.sidebar.date_input(
    "날짜 선택",
    value=default_date.date(),
    min_value=tmin,
    max_value=tmax,
)
date = pd.Timestamp(date)

# 영역 프리셋
preset = st.sidebar.selectbox(
    "영역 선택",
    [
        "전 지구",
        "동아시아(한국 포함)",
        "북서태평양(일본-한반도)",
        "북대서양(미 동부~유럽)",
        "남태평양(적도~30°S)",
    ],
    index=0,
)

bbox_dict = {
    "전 지구": None,
    "동아시아(한국 포함)": (5, 55, 105, 150),
    "북서태평양(일본-한반도)": (20, 55, 120, 170),
    "북대서양(미 동부~유럽)": (0, 70, -80, 20),
    "남태평양(적도~30°S)": (-30, 5, 140, -90),  # 날짜변경선 횡단 예시
}
bbox = bbox_dict[preset]

# 색상 범위
vabs = st.sidebar.slider("색상 범위 절대값 (±°C)", 2.0, 8.0, 5.0, 0.5)

# 투영
proj_name = st.sidebar.selectbox("투영(화면)", ["Robinson", "PlateCarree", "Mollweide"])
if proj_name == "Robinson":
    projection = ccrs.Robinson()
elif proj_name == "Mollweide":
    projection = ccrs.Mollweide()
else:
    projection = ccrs.PlateCarree()

# -----------------------------
# 데이터 로드 & 시각화
# -----------------------------
with st.spinner("SOEST ERDDAP에서 지도 데이터를 불러오는 중..."):
    try:
        da = load_anomaly(date, bbox=bbox)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

actual_date = pd.to_datetime(da["time"].values).date()
st.success(f"가져온 실제 날짜: {actual_date} (데이터 커버리지: {tmin} ~ {tmax})")

extent = None if bbox is None else (bbox[2], bbox[3], bbox[0], bbox[1])
title = f"OISST v2.1 해수면 온도 편차 (°C) · {preset} · {actual_date} · {proj_name}"

fig = plot_cartopy_anomaly(da, title, vabs=vabs, projection=projection, extent=extent)
st.pyplot(fig, clear_figure=True)

# -----------------------------
# 통계 & 다운로드
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("평균 편차 (°C)", f"{np.nanmean(da.values):+.2f}")
c2.metric("최대 편차 (°C)", f"{np.nanmax(da.values):+.2f}")
c3.metric("최소 편차 (°C)", f"{np.nanmin(da.values):+.2f}")

with st.expander("픽셀 데이터(샘플) 보기"):
    sample = da.coarsen(lat=4, lon=4, boundary="trim").mean()
    df_sample = sample.to_dataframe(name="anom(°C)").reset_index()
    # 🔑 NaN 값 제거
    df_sample = df_sample.dropna(subset=["anom(°C)"])
    st.dataframe(df_sample.head(200), use_container_width=True)

# 🔑 CSV도 NaN 제거
df_csv = da.to_dataframe(name="anom(°C)").reset_index()
df_csv = df_csv.dropna(subset=["anom(°C)"])

if df_csv.empty:
    st.warning("이 날짜/영역에는 유효한 anomaly 값이 없어 CSV가 비어 있습니다.")
else:
    csv_bytes = df_csv.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 현재 지도 데이터(CSV) 내려받기",
        data=csv_bytes,
        file_name=f"oisst_anom_{actual_date}_{preset}_{proj_name}.csv",
        mime="text/csv",
    )

# -----------------------------
# 📘 데이터 탐구 보고서 (학생용)
# -----------------------------
st.markdown("---")
st.header("📘 데이터 탐구 보고서: 우리 모둠의 발견")

st.subheader("1. 대한민국 주변 바다가 보여준 이상 신호")
st.markdown("""
2024년 8월 15일 기준 해수면 온도 편차 지도를 보면, 대한민국 주변 바다가 
세계적으로도 뚜렷한 **수온 상승의 핫스팟**으로 나타났습니다.  
동중국해, 대한해협, 동해 남부 해역 일대가 기준치보다 훨씬 높은 온도를 기록하며 
빨간색 영역으로 두드러졌습니다.  
이것은 우리 생활권과 직접 연결된 바다가 기후 위기의 최전선에 놓여 있음을 의미합니다.
""")

st.subheader("2. 해수온도 상승의 주요 원인")
st.markdown("""
첫째, **온실가스 배출 증가**로 인한 지구 온난화가 바다에 축적된 열을 키우고 있습니다.  
바다는 대기에서 발생한 초과 에너지의 90% 이상을 흡수하기 때문에, 
인간이 배출한 이산화탄소와 메탄이 결국 바다 온도를 밀어올리고 있습니다.  

둘째, **북태평양 해류와 대기 순환의 변화**가 한국 인근 해역을 특히 민감하게 만들었습니다.  
적도 부근에서 발생한 해양 열파(마린 히트웨이브)가 북상하면서 
한반도 주변 바다에 강한 온도 이상을 일으킨 것입니다.
""")

st.subheader("3. 해수온도 상승이 불러온 영향")
st.markdown("""
해수면 온도의 급격한 상승은 단순히 바닷물이 따뜻해지는 현상에 그치지 않습니다.  

- **어장 붕괴와 어종 이동**: 명태, 오징어 같은 냉수성 어종은 급격히 줄고, 
  대신 열대성 어종이 나타나며 어업 구조 자체가 변하고 있습니다.  

- **태풍의 위력 강화**: 따뜻한 바다는 태풍의 에너지원이 되기 때문에, 
  여름철 한반도를 향하는 태풍은 더욱 강력해지고 그 피해 규모도 커지고 있습니다.  

- **집중호우와 참사**: 바다에서 증발한 수증기가 많아질수록 
  대기 중 수분이 과도하게 축적되어 집중호우를 일으킵니다.  
  최근 우리나라에서 발생한 도시 침수, 산사태 같은 참사는 
  해수온도 상승과 무관하지 않으며, 이는 기후 위기가 
  인명 피해와 사회적 재난으로 직결되고 있음을 보여줍니다.  

- **연안 생태계 교란**: 해양 산성화와 함께, 산호 군락이나 해조류 숲 같은 
  연안 생태계가 무너지고 이는 다시 해양 생물 다양성 감소로 이어집니다.  

이러한 변화는 곧 우리의 식량, 안전, 지역 사회의 경제와 직결된다는 점에서 
단순히 환경 문제가 아닌 **생존의 문제**라고 할 수 있습니다.
""")


# -----------------------------
# 📚 참고자료
# -----------------------------
st.markdown("---")

st.markdown("""
### 📚 참고문헌

- NOAA National Centers for Environmental Information. (2019). *Optimum interpolation sea surface temperature (OISST) v2.1 daily high resolution dataset* [Data set]. NOAA National Centers for Environmental Information. https://www.ncei.noaa.gov/products/optimum-interpolation-sst  

- NOAA Atlantic Oceanographic and Meteorological Laboratory (AOML). (2025). *ERDDAP server: SST_OI_DAILY_1981_PRESENT_T (OISST v2.1, daily, 1981–present)* [Data set]. NOAA AOML. https://erddap.aoml.noaa.gov/hdb/erddap/info/SST_OI_DAILY_1981_PRESENT_T/index.html  

- 그레타 툰베리, 《기후 책》, 이순희 역, 기후변화행동연구소 감수, 열린책들, 2023.  
    ([Yes24 도서 정보 링크](https://www.yes24.com/product/goods/119700330))
""")



# -----------------------------
# Footer (팀명)
# -----------------------------
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: gray; font-size: 0.9em;'>
        화원고등학교 교사 최윤주
    </div>
    """,
    unsafe_allow_html=True
)