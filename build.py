import pandas as pd
import numpy as np

import os
from scipy import stats
import shutil
import pathlib
import jinja2
from jinja2 import Template

WINDOW_SIZE = 7
DISPLAY_WEEKS = 4
LOW_CUTOFF = 50


templateLoader = jinja2.FileSystemLoader(searchpath="./templates")
templateEnv = jinja2.Environment(loader=templateLoader)

import datetime as dt

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib import font_manager as fm

plt.style.use(['default', 'seaborn-poster', 'fivethirtyeight'])


font_dirs = ['./fonts', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for f in font_files:
    fm.fontManager.addfont(f)


df = pd.read_csv('covid-eucdc.csv', dtype={'geoId': str}, keep_default_na=False)

# Process dates in dateRep column.
df['dateRep'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')

# Build the country-lookup map.
country_lookup = {}
for country, geoId in df[['countriesAndTerritories', 'geoId']].values:
    if len(geoId) == 2:
        country_lookup[geoId] = country.replace('_', ' ')

# Lookup for dynamic map.
tla_country_lookup = {}
for tlcc, geoId in df[['countryterritoryCode', 'geoId']].values:
    if len(geoId) == 2:
        tla_country_lookup[geoId] = tlcc.replace('_', ' ')

# Fixes/additions
tla_country_lookup['CZ'] = 'CZE'



# Regroup by country code.
cdf = df.set_index('dateRep')
cdf = cdf[ ['cases', 'deaths', 'geoId']]
cdf = cdf.pivot(columns='geoId').reorder_levels([1,0], axis=1).sort_index(axis=1)
cdf = cdf.fillna(0)

# Calculate a 7-day rolling mean across the data to smooth.
crd = cdf.rolling(7, center=True, min_periods=1).mean()

# Calculate a 7-day rolling difference across the data, smooth the result.
cfc = crd.rolling(7).apply(lambda x: x.iloc[-1]-x.iloc[0])
cfc = cfc.fillna(0)
cfc = cfc.rolling(3, center=True, min_periods=1).mean()

###########
#### cfc is our final dataset, generate output.
###########

def plot(df, kind, country=None):
    
    if country:
        df = df[(country, kind)]
        
    else:
        df = df.xs(kind, level=1, axis=1, drop_level=False)
        df = df.sum(axis=1)

    # Default plot to last 28 days (ish)
    df = df[-7*DISPLAY_WEEKS:]  # display +1 week for initial smoothing

    plt.clf()
    ax = df.plot()
        
    ax.axhline(0, lw=3, c='k', ls='--')
    
    ylim_top = np.max(df.values) * 1.2
    ylim_bot = np.max(-df.values) * 1.2
    ylim_max = np.max([ylim_top, ylim_bot])

    if ylim_max > 0:
        ax.set_ylim(-np.mean([ylim_bot, ylim_bot, ylim_max]), np.mean([ylim_top, ylim_top, ylim_max]))
    
    ax.set_ylabel('Daily rate change vs. Previous week')
    ax.set_xlabel('')
    
    country_name = country_lookup[country] if country else 'World'
    title = ax.set_title('Change in rate of Coronavirus {} | {}'.format(kind, country_name))
    title.set_position([.5, 1.05])
    
    ax.fill_between(
        df.index.values, 
        df.values, 
        y2=0, 
        where=df.values > 0,
        interpolate=True, 
        fc='#d53e4f',
        alpha=0.8
    )

    ax.fill_between(
        df.index.values, 
        df.values, 
        y2=0, 
        where=df.values < 0,
        interpolate=True, 
        fc='#66c2a5', 
        alpha=0.8
    )
    
    ax.text(0.025, 0.95, 'INCREASING',
        bbox={'facecolor': '#d53e4f', 'alpha': 0.5}, transform=ax.transAxes)

    ax.text(0.025, 0.05, 'DECREASING',
        bbox={'facecolor': '#66c2a5', 'alpha': 0.5}, transform=ax.transAxes)

    return ax.figure


def is_it_better_yet(df, country=None):
    if country:
        df = df[country]
        
    else:
        df = df.sum(axis=1, level=1)

    total = df.sum()

    WINDOW_SIZE = 7

    index = np.array(df[-WINDOW_SIZE:].index.values, dtype=float)
    window = df[-WINDOW_SIZE:]

    slope_c, _, _, _, _ = stats.linregress(
        index, 
        window.cases
    )

    slope_d, _, _, _, _ = stats.linregress(
        index, 
        window.deaths
    )

    # Trend is based on slope ("early indicator") and latest data.
    trend = lambda: None
    trend.cases = slope_c <= 0 or (window.cases[-3:] > 0).sum() < 3
    trend.deaths = slope_d <= 0 or (window.deaths[-3:] > 0).sum() < 3

    # Absolute term is based on absolute drop.
    absolute = window.sum() <= 0

    indec = {
        True: 'down',
        False: 'up',
    }

    if total.cases < LOW_CUTOFF:
        return (
            "tbc",  # status when not enough data
            indec[trend.cases], indec[absolute.deaths],
            {
            'cases':"There are too few cases to get an accurate picture. But no news may be good news.",  # cases
            'deaths':"There are too few cases to get an accurate picture. But no news may be good news.",  # deaths
            'headline': "There are too few cases to get an accurate picture. But no news may be good news.", #statement
            }
        )



    status = {
        # 3 < 0, 7 < 0
        (False, False): 'no',
        (False, True): 'uh oh',
        (True, False): 'maybe',
        (True, True): 'yes'
    }[(trend.cases, absolute.cases)]

    statement_c = {
        # 3 < 0, 7 < 0
        (False, False): 'The number of daily cases is increasing day by day.',
        (False, True): 'There are signs of an acceleration in daily cases.',
        (True, False): 'There are signs of an deceleration in daily cases.',
        (True, True): 'The number of daily cases is falling.'
    }[(trend.cases, absolute.cases)]

    statement_d = {
        # 3 < 0, 7 < 0
        (False, False): 'The number of daily deaths is increasing day by day.',
        (False, True): 'There are signs of an acceleration in daily deaths.',
        (True, False): 'There are signs of an deceleration in daily deaths.',
        (True, True): 'The number of daily deaths is falling.'
    }[(trend.deaths, absolute.deaths)]    

    statement_h = {
        # cases 3 < 0, 7 < 0;; deaths 3 < 0; 7 < 0
        (False, False, False, False):'The number of daily cases and the number of daily deaths is still increasing.',
        (False, False, False, True): 'The number of daily cases is increasing. There are signs of an acceleration in daily deaths.',
        (False, False, True, False): 'The number of daily cases is increasing although there are signs of a deceleration in daily deaths.',
        (False, False, True, True): 'The number of daily cases is increasing although the number of daily deaths is decreasing.',

        (False, True, False, False): 'There are signs of an acceleration in daily cases, and daily deaths are increasing.',
        (False, True, False, True): 'There are signs of an acceleration in daily cases and deaths.',
        (False, True, True, False): 'There are signs of an acceleration in daily cases, but there are signs of a deceleration in daily deaths.',
        (False, True, True, True): 'There are signs of an acceleration in daily cases, but daily deaths continue to fall.',

        (True, False, False, False): 'There are signs of a deceleration in daily cases. However, daily deaths are still increasing.',
        (True, False, False, True): 'There are signs of a deceleration in daily cases, although there are signs of an acceleration in daily deaths.',
        (True, False, True, False): 'There are signs of a deceleration in daily cases and daily deaths.',
        (True, False, True, True): 'There are signs of a deceleration in daily cases, and daily deaths continue to fall.',

        (True, True, False, False): 'While the number of daily deaths is still increasing, the number of daily cases is decreasing.',
        (True, True, False, True): 'The number of daily cases is decreasing, but there are signs of an acceleration in daily deaths.',
        (True, True, True, False): 'The number of daily cases is decreasing, and there are signs of a deceleration in daily deaths.',
        (True, True, True, True): 'The number of daily cases and the number of daily deaths is decreasing.',
    }[(trend.cases, absolute.cases, trend.deaths, absolute.deaths)]

    statements = {
        'cases': statement_c,
        'deaths': statement_d,
        'headline': statement_h,
    }

    
    return status, indec[absolute.cases], indec[absolute.deaths], statements


def status_card(country_name, status):
    bg, fg = {
        'yes':   ('#66c2a5', '#ffffff'),
        'no':    ('#d53e4f', '#000000'),
        'uh oh': ('#863ed5', '#ffffff'),
        'maybe': ('#fee090', '#000000'),
        'tbc':   ('#eeeeee', '#000000'),
    }[status]

    fig = Figure(figsize=(8,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(bg)
    
    ax.text(
        0.5, 
        0.2, 
        'Is Coronavirus getting better in '.upper(), 
        fontname="Raleway",
        size=12, 
        color=fg, 
        ha='center',
        va='center',
        transform=ax.transAxes,
        alpha=0.8,
    )

    ax.text(
        0.5, 
        0.1, 
        '{}?'.format(country_name), 
        fontname="Raleway",
        size=24, 
        color=fg, 
        ha='center',
        va='center',
        transform=ax.transAxes,
        alpha=0.8,
    )

    ax.text(
        0.5, 
        0.5, 
        status.upper(), 
        size=100, 
        color=fg, 
        ha='center',
        va='center',
        weight='bold',
        transform=ax.transAxes,
        alpha=0.8,
    )
    
    return fig




# Generate the per-country html & plots.

country_status = {}  # Build a list of countries and statuses, for sorted homepage table.

template_c = templateEnv.get_template('country.html')



for country_id, country in country_lookup.items():

    #if country_id not in ['CN', 'NO', 'CZ', 'CH', 'CR', 'UY', 'LB','DO', 'SM', 'NL', 'IT', 'UK', 'ES', 'US', 'DE', 'MA', 'MU', 'ZA', 'AD']:
    #    continue
    
    print(country_lookup[country_id])

    country_path = os.path.join('build', country_id.lower())
    pathlib.Path(country_path).mkdir(parents=True, exist_ok=True)

    status, cases, deaths, statements = is_it_better_yet(cfc, country_id)

    fig = plot(cfc, 'cases', country_id)
    fig.savefig(os.path.join(country_path, 'cases.png'), bbox_inches="tight", pad_inches=0.5)

    fig = plot(cfc, 'deaths', country_id)
    fig.savefig(os.path.join(country_path, 'deaths.png'), bbox_inches="tight", pad_inches=0.5)
    
    fig = status_card(country_lookup[country_id], status)
    fig.savefig(os.path.join(country_path, 'card.png'), bbox_inches="tight")

    html = template_c.render(
        country_id=country_id,
        country=country,
        countries=country_lookup,
        status=status,
        statements=statements,
    )

    country_status[country_id] = {
        'status': status,
        'slug': status.replace(' ', '_'),
        'cases': cases,
        'deaths': deaths
    }

    with open(os.path.join(country_path, 'index.html'), 'w') as f:
        f.write(html)

## Do global plots and status

status, _, _, statements = is_it_better_yet(cfc)
fig = plot(cfc, 'cases')
fig.savefig(os.path.join('build', 'cases.png'), bbox_inches="tight", pad_inches=0.5)

fig = plot(cfc, 'deaths')
fig.savefig(os.path.join('build', 'deaths.png'), bbox_inches="tight", pad_inches=0.5)

fig = status_card('The World', status)
fig.savefig(os.path.join('build', 'card.png'), bbox_inches="tight")

template_h = templateEnv.get_template('home.html')

html = template_h.render(
    country_id='WORLD',
    country="World",
    countries=country_lookup,
    country_status=country_status,
    tlacountries=tla_country_lookup,
    status=status,
    statements=statements,
)

with open(os.path.join('build', 'index.html'), 'w') as f:
    f.write(html)

sitemap = templateEnv.get_template('sitemap.xml')

xml = sitemap.render(
    countries=country_lookup,
    today=dt.datetime.today()
)

with open(os.path.join('build', 'sitemap.xml'), 'w') as f:
    f.write(xml)

