############################################################################################
#                                  Author: Anass MAJJI                                     #
#                               File Name: streamlit_app.py                                #
#                           Creation Date: November 06, 2022                               #
#                         Source Language: Python                                          #
#         Repository:    https://github.com/amajji/Streamlit-Dash.git                      #
#                              --- Code Description ---                                    #
#         Streamlit app designed for visualizing U.S. real estate data and market trends   #
############################################################################################


############################################################################################
#                                   Packages                                               #
############################################################################################


# Import Python Libraries
import pandas as pd
import folium
import geopandas as gpd
from folium.features import GeoJsonPopup, GeoJsonTooltip
import streamlit as st
from streamlit_folium import folium_static
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import pgeocode
import numpy as np
import pydeck as pdk
import zipfile
import geopandas as gpd
import leafmap.colormaps as cm
from leafmap.common import hex_to_rgb
from uszipcode import SearchEngine
from random import randint
from multiapp import MultiApp
import pathlib

#########################################################################################
#                                Variables                                              #
#########################################################################################

st.set_page_config(layout="wide")
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / "static"
# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

global df_final
global gdf


#########################################################################################
#                                Functions                                              #
#########################################################################################


def sidebar_caption():
    """This is a demo of some shared sidebar elements.
    Reused this function to make sure we have the same sidebar elements if needed.
    """

    # st.sidebar.header("Yay, this is a sidebar")
    # st.sidebar.markdown("")


@st.cache(allow_output_mutation=True)
def read_xlsx(path):
    # read each excel file
    excel_file = pd.ExcelFile(path)

    list_columns = [
        "Down Payment Source",
        "Loan Purpose",
        "Property Type",
        "Property State",
        "Property City",
        "Property Zip",
        "Interest Rate",
    ]

    purchase_sheet_name = "Purchase Data April 2018"
    refinance_sheet_name = "Refinance Data April 2018"

    df_purshase = excel_file.parse(purchase_sheet_name)
    df_purshase = df_purshase[list_columns]

    df_refinance = excel_file.parse(refinance_sheet_name)
    df_refinance = df_refinance[list_columns]

    return df_purshase.append(df_refinance)


@st.cache
def read_file(path):
    return gpd.read_file(path)


@st.cache
def get_geom_data(category):

    prefix = (
        "https://raw.githubusercontent.com/giswqs/streamlit-geospatial/master/data/"
    )
    links = {
        "national": prefix + "us_nation.geojson",
        "state": prefix + "us_states.geojson",
        "county": prefix + "us_counties.geojson",
        "metro": prefix + "us_metro_areas.geojson",
        "zip": "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_zcta510_500k.zip",
    }

    if category.lower() == "zip":
        r = requests.get(links[category])
        out_zip = os.path.join(DOWNLOADS_PATH, "cb_2018_us_zcta510_500k.zip")
        with open(out_zip, "wb") as code:
            code.write(r.content)
        zip_ref = zipfile.ZipFile(out_zip, "r")
        zip_ref.extractall(DOWNLOADS_PATH)
        gdf = gpd.read_file(out_zip.replace("zip", "shp"))
    else:
        gdf = gpd.read_file(links[category])
    return gdf


@st.cache
def join_attributes(gdf, df, category):

    new_gdf = None
    if category == "county":
        gdf["GEOID10"] = gdf["GEOID10"].astype(int)
        df["Property Zip"] = df["Property Zip"].astype(int)
        new_gdf = gdf.merge(df, left_on="GEOID10", right_on="Property Zip", how="inner")
    elif category == "state":
        new_gdf = gdf.merge(df, left_on="STUSPS", right_on="STUSPS", how="outer")
    elif category == "national":
        if "geo_country" in df.columns.values.tolist():
            df["country"] = None
            df.loc[0, "country"] = "United States"
        new_gdf = gdf.merge(df, left_on="NAME", right_on="country", how="outer")
    elif category == "metro":
        new_gdf = gdf.merge(df, left_on="CBSAFP", right_on="cbsa_code", how="outer")
    elif category == "zip":
        new_gdf = gdf.merge(df, left_on="GEOID10", right_on="postal_code", how="outer")
    return new_gdf


@st.cache(allow_output_mutation=True)
def get_loan_data(scale, df_final, type_loan, column_selected):

    if type_loan == "Purchase":
        df_temp = df_final[df_final["Loan Purpose"] == "Purchase"]
    else:
        df_temp = df_final[df_final["Loan Purpose"] != "Purchase"]

    df_temp["Property Zip"] = df_temp["Property Zip"].astype(int)
    df_temp["Number of loans granted"] = 1

    df_gpby_nbr_loans = (
        df_temp.groupby("Property Zip")
        .sum()
        .reset_index()[["Property Zip", "Number of loans granted"]]
    )
    df_gpby_interest_rate = (
        df_temp.groupby("Property Zip")
        .mean()
        .reset_index()[["Property Zip", "Interest Rate"]]
    )

    df_grby_nbr_loan_IR = pd.merge(
        df_gpby_nbr_loans,
        df_gpby_interest_rate,
        left_on="Property Zip",
        right_on="Property Zip",
        how="inner",
    )

    if scale == "State":

        engine = SearchEngine()
        df_grby_nbr_loan_IR["state"] = df_grby_nbr_loan_IR["Property Zip"].apply(
            lambda x: engine.by_zipcode(int(x)).state
        )
        df_grby_nbr_loan_IR["state"] = df_grby_nbr_loan_IR["state"].astype(str)

        if column_selected == "Interest Rate":
            df_grby_nbr_loan_IR = (
                df_grby_nbr_loan_IR.groupby("state")
                .mean("Interest Rate")
                .reset_index()[["state", "Interest Rate"]]
            )

        else:
            df_grby_nbr_loan_IR = (
                df_grby_nbr_loan_IR.groupby("state")
                .sum("Number of loans granted")
                .reset_index()[["state", "Number of loans granted"]]
            )

        gdf = get_geom_data("state")
        gdf = pd.merge(df_grby_nbr_loan_IR, gdf, left_on="state", right_on="STUSPS")
        gdf = gpd.GeoDataFrame(gdf, crs="EPSG:4326", geometry="geometry")

    else:

        if column_selected == "Interest Rate":
            df_grby_nbr_loan_IR = (
                df_grby_nbr_loan_IR.groupby("Property Zip")
                .mean("Interest Rate")
                .reset_index()[["Property Zip", "Interest Rate"]]
            )

        else:
            df_grby_nbr_loan_IR = (
                df_grby_nbr_loan_IR.groupby("Property Zip")
                .sum("Number of loans granted")
                .reset_index()[["Property Zip", "Number of loans granted"]]
            )

        gdf = gpd.read_file(
            STREAMLIT_STATIC_PATH + "/cb_2018_us_zcta510_500k.shp"
        )
        gdf = join_attributes(gdf, df_grby_nbr_loan_IR, scale.lower())
        # gdf["count"] = gdf["count"].astype(int)

    return gdf.iloc[:2000]


def select_non_null(gdf, col_name):
    new_gdf = gdf[~gdf[col_name].isna()]
    return new_gdf


def select_null(gdf, col_name):
    new_gdf = gdf[gdf[col_name].isna()]
    return new_gdf


#########################################################################################
#                                Main code                                              #
#########################################################################################


# Read DataFrame
df_final = read_xlsx(
    STREAMLIT_STATIC_PATH + "/snap_2018.xlsx"
)


# Read the geojson file
gdf = read_file(
    STREAMLIT_STATIC_PATH + "/us-state-boundaries.geojson"
)


# First page
def page_1():

    st.title("✨ U.S. Real Estate Data and Market Trends")
    st.markdown(
        """This interactive dashboard is designed for visualizing U.S. real estate data and market trends at multiple levels (i.e. state, county). The data sources include [Real Estate Data](https://www.hud.gov/program_offices/housing/rmra/oe/rpts/sfsnap/sfsnap) from the US Department of Housing and Urban Development’s website and 
         [Cartographic Boundary Files](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html) from U.S. Census Bureau.
         Several open-source packages are used to process the data and generate the visualizations, e.g., [streamlit](https://streamlit.io),
          [geopandas](https://geopandas.org), [leafmap](https://leafmap.org), [matplotlib](https://matplotlib.org/) and [pydeck](https://deckgl.readthedocs.io).
    """
    )

    st.markdown("# Data Analysis")
    st.markdown(
        """Below, we show the variation of some features related to the purchase and refinance loans, in particular Down Payment Source, Property Type, Property State and Interest Rate.
    """
    )

    # Create three columns/filters
    col1, col2 = st.columns(2)

    with col1:
        state_list = df_final["Property State"].unique().tolist()
        state_list.sort(reverse=True)
        state = st.selectbox("Property State", state_list, index=0)

    with col2:
        zip_list = df_final["Property Zip"].unique().tolist()
        zip_list.sort(reverse=True)
        zip_value = st.selectbox("Property Zip", zip_list, index=0)

    if st.checkbox("Show raw data"):
        st.subheader("Raw data")
        st.write(df_final)

    st.markdown("## Purchase and refiance classes")
    df_final.loc[df_final["Loan Purpose"] != "Purchase", "Loan Purpose"] = "refinance"
    df1 = df_final["Loan Purpose"].value_counts().rename_axis("unique_values")

    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:

        col2.bar_chart(df1, width=600, height=600)

    with col3:
        st.write("")

    st.markdown("## Down Payment Source")

    col1, col2 = st.columns(2)
    col1.markdown("### Purchase")
    col2.markdown("### Refinance")

    df_final["count"] = 1
    df_frequence = (
        df_final[df_final["Loan Purpose"] == "Purchase"]
        .groupby("Down Payment Source")
        .count()
    )
    df_frequence.sort_values(by="count", ascending=False, inplace=True)

    # create data
    names = df_frequence.index
    size = df_frequence["count"]

    # Create a circle at the center of the plot
    my_circle = plt.Circle((0, 0), 0.7, color="white")

    # Custom wedges
    fig_pie = plt.pie(
        size, labels=names, wedgeprops={"linewidth": 9, "edgecolor": "white"}
    )
    p = plt.gcf()
    p.set_size_inches(6, 6)
    p.gca().add_artist(my_circle)

    col1.pyplot(p)

    df_final["count"] = 1
    df_frequence = (
        df_final[df_final["Loan Purpose"] != "Purchase"]
        .groupby("Down Payment Source")
        .count()
    )
    df_frequence.sort_values(by="count", ascending=False, inplace=True)

    # create data
    names = df_frequence.index
    size = df_frequence["count"]

    # Create a circle at the center of the plot
    my_circle = plt.Circle((0, 0), 0.7, color="white")

    # Custom wedges
    fig_pie2 = plt.pie(
        size, labels=names, wedgeprops={"linewidth": 9, "edgecolor": "white"}
    )
    p = plt.gcf()
    p.set_size_inches(6, 6)
    p.gca().add_artist(my_circle)

    col2.pyplot(p)

    col1_comment, col2_comment, col3_comment = st.columns([1, 6, 1])

    with col1_comment:
        st.write("")

    with col2_comment:
        col2_comment.markdown(
            "This figure shows the distribution of Down Payment Source feature: Buyers mostly borrow the down payment or get it from their relatives, It means they don't have enough cash."
        )

    with col3_comment:
        st.write("")

    col3, col4 = st.columns(2)
    df_final["Interest Rate"] = df_final["Interest Rate"].astype(float)

    f1 = plt.figure(figsize=(6, 6))
    fig1 = sns.boxplot(
        x="Down Payment Source",
        y="Interest Rate",
        data=df_final[df_final["Loan Purpose"] == "Purchase"],
    )
    # fig1.axis()
    col3.pyplot(f1)

    f2 = plt.figure(figsize=(6, 6))
    fig2 = sns.boxplot(
        x="Down Payment Source",
        y="Interest Rate",
        data=df_final[df_final["Loan Purpose"] != "Purchase"],
    )
    # fig2.axis()
    col4.pyplot(f2)

    st.markdown("## Property Type")

    col5, col6 = st.columns(2)
    col5.markdown("### Purchase")
    col6.markdown("### Refinance")

    df_frequence = (
        df_final[df_final["Loan Purpose"] == "Purchase"]
        .groupby("Property Type")
        .count()
    )
    df_frequence.sort_values(by="count", ascending=False, inplace=True)

    # create data
    names = df_frequence.index
    size = df_frequence["count"]

    # Create a circle at the center of the plot
    my_circle1 = plt.Circle((0, 0), 0.7, color="white")

    # Custom wedges
    plt.pie(size, labels=names, wedgeprops={"linewidth": 7, "edgecolor": "white"})
    p2 = plt.gcf()
    p2.set_size_inches(10, 10)
    p2.gca().add_artist(my_circle1)
    col5.pyplot(p2)

    df_frequence = (
        df_final[df_final["Loan Purpose"] != "Purchase"]
        .groupby("Property Type")
        .count()
    )
    df_frequence.sort_values(by="count", ascending=False, inplace=True)

    # create data
    names = df_frequence.index
    size = df_frequence["count"]

    # Create a circle at the center of the plot
    my_circle3 = plt.Circle((0, 0), 0.7, color="white")

    # Custom wedges
    plt.pie(size, labels=names, wedgeprops={"linewidth": 7, "edgecolor": "white"})
    p3 = plt.gcf()
    p3.set_size_inches(10, 10)
    p3.gca().add_artist(my_circle3)
    col6.pyplot(p3)

    col7, col8 = st.columns(2)

    f = plt.figure(figsize=(6, 6))
    fig = sns.boxplot(
        x="Property Type",
        y="Interest Rate",
        data=df_final[df_final["Loan Purpose"] == "Purchase"],
    )
    fig.axis()
    col7.pyplot(f)

    f = plt.figure(figsize=(6, 6))
    fig = sns.boxplot(
        x="Property Type",
        y="Interest Rate",
        data=df_final[df_final["Loan Purpose"] != "Purchase"],
    )
    fig.axis()
    col8.pyplot(f)

    st.markdown("## Property State")

    col9, col10 = st.columns(2)
    col9.markdown("### Purchase")
    col10.markdown("### Refinance")

    df_frequence = (
        df_final[df_final["Loan Purpose"] == "Purchase"]
        .groupby("Property State")
        .count()
    )
    df_frequence["count"] = df_frequence["count"].apply(
        lambda x: x / df_frequence["count"].sum()
    )
    df_frequence = df_frequence[df_frequence["count"] > 0.008]
    df_frequence.sort_values(by="count", ascending=False, inplace=True)

    # create data
    names = df_frequence.index
    size = df_frequence["count"]

    # Create a circle at the center of the plot
    my_circle = plt.Circle((0, 0), 0.7, color="white")

    # Custom wedges
    plt.pie(size, labels=names, wedgeprops={"linewidth": 7, "edgecolor": "white"})
    p = plt.gcf()
    p.set_size_inches(10, 10)
    p.gca().add_artist(my_circle)
    col9.pyplot(p)

    df_frequence = (
        df_final[df_final["Loan Purpose"] == "Purchase"]
        .groupby("Property State")
        .count()
    )
    df_frequence["count"] = df_frequence["count"].apply(
        lambda x: x / df_frequence["count"].sum()
    )
    df_frequence = df_frequence[df_frequence["count"] > 0.008]
    df_frequence.sort_values(by="count", ascending=False, inplace=True)

    # create data
    names = df_frequence.index
    size = df_frequence["count"]

    # Create a circle at the center of the plot
    my_circle = plt.Circle((0, 0), 0.7, color="white")

    # Custom wedges
    plt.pie(size, labels=names, wedgeprops={"linewidth": 7, "edgecolor": "white"})
    p = plt.gcf()
    p.set_size_inches(10, 10)
    p.gca().add_artist(my_circle)
    col10.pyplot(p)

    col11, col12 = st.columns(2)

    f, ax = plt.subplots(figsize=(12, 12))
    fig = sns.boxplot(
        x="Property State",
        y="Interest Rate",
        data=df_final[df_final["Loan Purpose"] == "Purchase"][
            df_final[df_final["Loan Purpose"] == "Purchase"]["Property State"].isin(
                df_frequence.index.tolist()
            )
        ],
    )
    fig.axis()
    col11.pyplot(f)

    f, ax = plt.subplots(figsize=(12, 12))
    fig = sns.boxplot(
        x="Property State",
        y="Interest Rate",
        data=df_final[df_final["Loan Purpose"] != "Purchase"][
            df_final[df_final["Loan Purpose"] != "Purchase"]["Property State"].isin(
                df_frequence.index.tolist()
            )
        ],
    )
    fig.axis()
    col12.pyplot(f)

    df_final_temp = df_final.iloc[:1000]

    st.markdown("## Interest Rate")
    col13, col14 = st.columns(2)

    col13.markdown("### Purchase")
    col14.markdown("### Refinance")

    df_purchase_temp = pd.DataFrame(
        {
            "x_values": range(
                1,
                len(df_final[df_final["Loan Purpose"] == "Purchase"]["Interest Rate"])
                + 1,
            ),
            "y_values": df_final[df_final["Loan Purpose"] == "Purchase"][
                "Interest Rate"
            ].tolist(),
        }
    )
    df_refinance_temp = pd.DataFrame(
        {
            "x_values": range(
                1,
                len(df_final[df_final["Loan Purpose"] != "Purchase"]["Interest Rate"])
                + 1,
            ),
            "y_values": df_final[df_final["Loan Purpose"] != "Purchase"][
                "Interest Rate"
            ].tolist(),
        }
    )

    # set the figure size
    fig_purchase = plt.figure(figsize=(15, 10))

    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
    sns.set(style="darkgrid")

    # Make boxplot for one group only
    sns.violinplot(y=df_purchase_temp["y_values"])
    col13.pyplot(fig_purchase)

    fig_refinance = plt.figure(figsize=(15, 10))
    # Make boxplot for one group only
    sns.violinplot(y=df_refinance_temp["y_values"])
    col14.pyplot(fig_refinance)


# Second page
def page_2():

    st.title("✨ U.S. Real Estate Data and Market Trends")
    st.markdown(
        """This interactive dashboard is designed for visualizing U.S. real estate data and market trends at multiple levels (i.e. state, county). The data sources include [Real Estate Data](https://www.hud.gov/program_offices/housing/rmra/oe/rpts/sfsnap/sfsnap) from the US Department of Housing and Urban Development’s website and 
         [Cartographic Boundary Files](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html) from U.S. Census Bureau.
         Several open-source packages are used to process the data and generate the visualizations, e.g., [streamlit](https://streamlit.io),
          [geopandas](https://geopandas.org), [leafmap](https://leafmap.org), and [pydeck](https://deckgl.readthedocs.io).
    """
    )

    # with st.expander("See a demo"):
    #    st.image("https://i.imgur.com/Z3dk6Tr.gif")

    scale = st.selectbox("Scale", ["State", "County"], index=0)
    type_loan = st.selectbox("Loan category", ["Purchase", "Refinance"], index=0)
    selected_col = st.selectbox(
        "Column", ["Number of loans granted", "Interest Rate"], index=0
    )

    gdf = get_loan_data(scale, df_final, type_loan, selected_col)
    gdf_null = select_null(gdf, selected_col)
    gdf = select_non_null(gdf, selected_col)
    gdf = gdf.sort_values(by=selected_col, ascending=True)

    row2_col1, row2_col2, row2_col3, row2_col4, row2_col5, row2_col6 = st.columns(
        [0.6, 0.68, 0.7, 0.7, 1.5, 0.8]
    )

    palettes = cm.list_colormaps()
    with row2_col1:
        palette = st.selectbox("Color palette", palettes, index=palettes.index("Blues"))
    with row2_col2:
        n_colors = st.slider("Number of colors", min_value=2, max_value=20, value=8)
    with row2_col3:
        show_nodata = st.checkbox("Show nodata areas", value=True)
    with row2_col4:
        show_3d = st.checkbox("Show 3D view", value=True)
    with row2_col5:
        if show_3d:
            elev_scale = st.slider(
                "Elevation scale", min_value=1, max_value=1000000, value=4, step=10
            )

            with row2_col6:
                st.info("Press Ctrl and move the left mouse button.")
        else:
            elev_scale = 1

    elev_scale = 200
    colors = cm.get_palette(palette, n_colors)
    colors = [hex_to_rgb(c) for c in colors]

    for i, ind in enumerate(gdf.index):
        index = int(i / (len(gdf) / len(colors)))
        if index >= len(colors):
            index = len(colors) - 1
        gdf.loc[ind, "R"] = colors[index][0]
        gdf.loc[ind, "G"] = colors[index][1]
        gdf.loc[ind, "B"] = colors[index][2]

    initial_view_state = pdk.ViewState(
        latitude=40,
        longitude=-100,
        zoom=3,
        max_zoom=16,
        pitch=0,
        bearing=0,
        height=900,
        width=None,
    )

    min_value = gdf[selected_col].min()
    max_value = gdf[selected_col].max()
    color = "color"
    # color_exp = f"[({selected_col}-{min_value})/({max_value}-{min_value})*255, 0, 0]"
    color_exp = f"[R, G, B]"

    geojson = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        pickable=True,
        opacity=0.5,
        stroked=True,
        filled=True,
        extruded=show_3d,
        wireframe=True,
        get_elevation=f"{selected_col}",
        elevation_scale=elev_scale,
        # get_fill_color="color",
        get_fill_color=color_exp,
        get_line_color=[0, 0, 0],
        get_line_width=2,
        line_width_min_pixels=1,
    )

    geojson_null = pdk.Layer(
        "GeoJsonLayer",
        gdf_null,
        pickable=True,
        opacity=0.2,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        # get_elevation="properties.ALAND/100000",
        # get_fill_color="color",
        get_fill_color=[200, 200, 200],
        get_line_color=[0, 0, 0],
        get_line_width=2,
        line_width_min_pixels=1,
    )

    # tooltip = {"text": "Name: {NAME}"}

    # tooltip_value = f"<b>Value:</b> {median_listing_price}""
    tooltip = {
        "html": "<b>Name:</b> {NAME}<br><b>Value:</b> {"
        + selected_col
        + "}<br><b>Date:</b> "
        + "",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    layers = [geojson]
    if show_nodata:
        layers.append(geojson_null)

    r = pdk.Deck(
        layers=layers,
        initial_view_state=initial_view_state,
        map_style="light",
        tooltip=tooltip,
    )

    row3_col1, row3_col2 = st.columns([6, 1])

    with row3_col1:
        st.pydeck_chart(r)
    with row3_col2:
        st.write(
            cm.create_colormap(
                palette,
                label=selected_col.replace("_", " ").title(),
                width=0.2,
                height=3,
                orientation="vertical",
                vmin=min_value,
                vmax=max_value,
                font_size=10,
            )
        )


def main():
    """A streamlit app template"""

    st.sidebar.title("Menu")

    PAGES = {"📊 Visualizations": page_1, "🌎 Interactive map": page_2}

    # Select pages
    # Use dropdown if you prefer
    selection = st.sidebar.radio("Select your page : ", list(PAGES.keys()))
    sidebar_caption()

    PAGES[selection]()

    st.sidebar.title("About")
    st.sidebar.info(
        """
    Web App URL: <https://anasma.streamlitapp.com>
    GitHub repository: <https://github.com/giswqs/streamlit-geospatial>
    """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
    MAJJI Anass 
    [GitHub](https://github.com/amajji) | [LinkedIn](https://fr.linkedin.com/in/anass-majji-729773157)
    """
    )


if __name__ == "__main__":
    main()
