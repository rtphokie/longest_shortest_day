import json
import time
import unittest
from math import modf
from pprint import pprint

from pytz import timezone
from skyfield import api, almanac
from skyfield.searchlib import find_maxima, find_minima
import datetime, pytz
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests_cache

ts = api.load.timescale()
load = api.Loader('/var/data')

s = requests_cache.CachedSession('tzapicache')
e = load('de430t.bsp')
earth, sun = e['earth'], e['sun']

now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)


def equilux(lat, lon, tzstr, year=2022, years=1):
    tz = timezone(tzstr)

    t0 = ts.utc(year, 1, 1)
    t1 = ts.utc(year, 12, 31)
    t, y = almanac.find_discrete(t0, t1, almanac.seasons(e))

    results = {}
    for yi, ti in zip(y, t):
        if 'Solstice' in almanac.SEASON_EVENTS[yi]:
            results[almanac.SEASON_EVENTS[yi]], _ = ti.astimezone_and_leap_second(tz)
        if 'Equinox' in almanac.SEASON_EVENTS[yi]:
            dt = ti.utc_datetime()
            results[almanac.SEASON_EVENTS[yi]], _ = ti.astimezone_and_leap_second(tz)
            observer = api.Topos(lat, lon)
            t0 = ts.utc(dt.year, dt.month, dt.day - 7)
            t1 = ts.utc(dt.year, dt.month, dt.day + 7)
            t2, y2 = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(e, observer))
            prev = None
            dates = []
            delta = []
            sunlight = []
            for t2i, y2i in zip(t2, y2):
                dt2, _ = t2i.astimezone_and_leap_second(tz)
                hrs = dt2.hour + dt2.minute / 60.0 + dt2.second / 3600.0
                if y2i:  # rise
                    prev = hrs
                elif prev is not None:  # set
                    dates.append(dt2)
                    delta.append(abs(hrs - prev - 12.0))
                    sunlight.append(hrs - prev)
            idx = delta.index(min(delta))
            results[almanac.SEASON_EVENTS[yi]], _ = ti.astimezone_and_leap_second(tz)

            hrs = int(sunlight[idx])
            rem = (sunlight[idx] - hrs) * 60.0
            mins = int(rem)
            secs = (rem - mins) * 60.0

            delta = 12 - sunlight[idx]
            if delta > 0:
                deltastr = f"{delta * 3600:.1f} seconds shy of"
            else:
                deltastr = f"{abs(delta * 3600):.1f} seconds more than"

            results[almanac.SEASON_EVENTS[yi].replace('nox', 'lux')] = dates[idx]
            results[f"{almanac.SEASON_EVENTS[yi].replace('nox', 'lux')} time"] = f'{hrs} hrs {mins} min {secs:.0f} sec'
            results[f"{almanac.SEASON_EVENTS[yi].replace('nox', 'lux')} delta"] = deltastr
            results[f"{almanac.SEASON_EVENTS[yi].replace('nox', 'lux')} hours"] = sunlight[idx]

    return results


def sun_distance_at(t):
    d = earth.at(t).observe(sun).apparent().distance()
    return d.km


def sundistance(year, tzstr):
    tz = timezone(tzstr)

    t0 = ts.utc(year, 1, 1)
    t1 = ts.utc(year + 1, 1, 1)
    sun_distance_at.step_days = 1

    aphelion = (None, None)
    perihelion = (None, None)

    times, distances = find_maxima(t0, t1, sun_distance_at)
    for t, distance in zip(times, distances):
        dt, _ = t.astimezone_and_leap_second(tz)
        aphelion = (dt, distance)

    times, distances = find_minima(t0, t1, sun_distance_at)
    for t, distance in zip(times, distances):
        dt, _ = t.astimezone_and_leap_second(tz)
        perihelion = (dt, distance)

    return perihelion, aphelion


def chart_rise_set(start, end, lat, lon, tzstr, cityname):
    startYear = int(start[:4])
    endYear = int(end[:4])
    extremes, data = risesetextremes(lat, lon, tzstr, startYear=startYear,
                                     years=endYear - startYear + 1, verbose=False)
    # pprint(extremes['2024'])
    # return
    print(f"{cityname} {extremes['2024']['rise_latest'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{extremes['2024']['Winter Solstice'].strftime('%Y-%m-%d %H:%M:%S %Z')}")

    dates_to_label = []
    milestones = []
    equinox_or_solstice = []
    result = {'dates': [],
              'rise_hrs': [], 'set_hrs': [],
              'rise_hrs_perm_dst': [], 'set_hrs_perm_dst': [],
              'daylight_hrs': [], 'betweensolarnoons': []}
    for year, data_year in data.items():
        for k, v in extremes[year].items():
            if type(v) is datetime.datetime:
                if 'Equinox' in k or 'Solstice' in k:
                    equinox_or_solstice.append(v.strftime('%Y-%m-%d'))
                    milestones.append(v.strftime('%Y-%m-%d'))

        for date, rise_hr, set_hr, rise_dt, betweensolarnoons in zip(data_year['dates'],
                                                                     data_year['rise_hr'],
                                                                     data_year['set_hr'],
                                                                     data_year['rises'],
                                                                     data_year['betweensolarnoons'],
                                                                     ):
            if date >= start and date <= end:
                result['dates'].append(date)
                if date in milestones:
                    dates_to_label.append(date)

                if rise_dt.strftime('%Z')[1] != 'D':
                    # simulate permanent daylight saving time
                    result['rise_hrs_perm_dst'].append(rise_hr + 1)
                    result['set_hrs_perm_dst'].append(set_hr + 1)
                else:
                    result['rise_hrs_perm_dst'].append(rise_hr)
                    result['set_hrs_perm_dst'].append(set_hr)
                result['rise_hrs'].append(rise_hr)
                result['set_hrs'].append(set_hr)
                result['betweensolarnoons'].append(set_hr)
                result['daylight_hrs'].append(set_hr - rise_hr)

    # Create a sample DataFrame
    df = pd.DataFrame(result)
    for dstperm in ['', '_perm_dst']:
        print(dstperm)
        for riseset in ['rise', 'set']:
            max_riseset = df.loc[df[f'{riseset}_hrs{dstperm}'].idxmax()]
            min_riseset = df.loc[df[f'{riseset}_hrs{dstperm}'].idxmin()]
            print(f"{riseset:4} min {min_riseset.dates} {min_riseset[f'{riseset}_hrs{dstperm}']:.2f}")
            print(f"{riseset:4} max {max_riseset.dates} {max_riseset[f'{riseset}_hrs{dstperm}']:.2f}")
        print()

    # Convert the 'date' column to datetime format
    df['dates'] = pd.to_datetime(df['dates'])

    # Create the line graph
    fig, ax = plt.subplots(figsize=(16, 9))
    df.plot(ax=ax, x='dates', y=['rise_hrs', 'set_hrs'], legend=False, lw=4, color=['g', 'r'])
    df.plot(ax=ax, x='dates', y=['rise_hrs_perm_dst', 'set_hrs_perm_dst'], legend=False, lw=4, color=['g', 'r'],
            alpha=.5, linestyle='dashed')

    date_format = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xlim([pd.to_datetime(start), pd.to_datetime(end)])

    ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))

    ax.set_xticks(pd.to_datetime(dates_to_label))
    plt.xticks(rotation=45)
    for date in equinox_or_solstice:
        ax.axvline(pd.to_datetime(date), color='k', linestyle='dotted', alpha=0.5, lw=1)
    # ax.set_title(f'Sunrise and Sunset Hours {cityname} {start} to {end}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig(f"sunriseset_{cityname.replace(' ', '_')}_{start}_{end}")


def chart_solar_days(start, end, lat, lon, tzstr, cityname):
    startYear = int(start[:4])
    endYear = int(end[:4])
    extremes, data = risesetextremes(lat, lon, tzstr, startYear=startYear,
                                     years=endYear - startYear + 1, verbose=False)
    # pprint(extremes['2024'])
    # return
    print(f"{cityname} {extremes['2024']['rise_latest'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{extremes['2024']['Winter Solstice'].strftime('%Y-%m-%d %H:%M:%S %Z')}")

    dates_to_label = []
    milestones = []
    equinox_or_solstice = []
    result = {'dates': [], 'betweensolarnoons': []}
    for year, data_year in data.items():
        for k, v in extremes[year].items():
            if type(v) is datetime.datetime:
                if 'helion' in k :
                    milestones.append(v.strftime('%Y-%m-%d'))

        for date, betweensolarnoons in zip(data_year['dates'], data_year['betweensolarnoons']):
            if date >= start and date <= end:
                result['dates'].append(date)
                if date in milestones:
                    dates_to_label.append(date)
                result['betweensolarnoons'].append(betweensolarnoons)

    # Create a sample DataFrame
    df = pd.DataFrame(result)
    df['delta']=60*60*(df['betweensolarnoons']-24.0)
    max_solarday = df.loc[df['betweensolarnoons'].idxmax()]
    min_solarday = df.loc[df['betweensolarnoons'].idxmin()]
    max_delta_hrs = max_solarday['betweensolarnoons'] - 24.0
    max_delta_min = max_delta_hrs*60
    max_delta_sec = max_delta_min*60
    min_delta_hrs = 24.0 - min_solarday['betweensolarnoons']
    min_delta_min = min_delta_hrs*60
    min_delta_sec = min_delta_min*60

    print(f"max {max_solarday.dates} {max_solarday['betweensolarnoons']:.4f} {max_delta_sec:.4f} seconds")
    print(f"min {min_solarday.dates} {min_solarday['betweensolarnoons']:.4f} {min_delta_sec:.4f} seconds")

    # Convert the 'date' column to datetime format
    df['dates'] = pd.to_datetime(df['dates'])

    # Create the line graph
    fig, ax = plt.subplots(figsize=(16, 9))
    df.plot(ax=ax, x='dates', y=['delta'], legend=False, lw=4, color=['k'])

    date_format = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xlim([pd.to_datetime(start), pd.to_datetime(end)])

    ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))
    ax.tick_params(axis='y', labelsize=24)

    ax.set_xticks(pd.to_datetime(dates_to_label))
    plt.xticks(rotation=45)
    for date in milestones:
        ax.axvline(pd.to_datetime(date), color='k', linestyle='dotted', alpha=0.5, lw=1)
    # ax.set_title(f'Sunrise and Sunset Hours {cityname} {start} to {end}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig(f"solardays_{cityname.replace(' ', '_')}_{start}_{end}")


def daily_solar_metrics(lat, lon, tzstr, startYear, years, verbose=False):
    # ts = api.load.timescale()
    # load = api.Loader('/var/data')

    tz = timezone(tzstr)

    observer = api.Topos(lat, lon)
    t0 = ts.utc(startYear, 1, 1)
    t1 = ts.utc(startYear + years, 1, 1)
    t, y = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(e, observer))
    f = almanac.meridian_transits(e, e['Sun'], observer)
    tm, te = almanac.find_discrete(t0, t1, f)
    solarnoons = []
    for tmi, tei in zip(tm, te):
        if tei == 1:
            solarnoons.append(tmi)

    result = dict()
    prevrise = None
    prevsolarnoon = None
    solarnoon = None
    riseset = ['set', 'rise']
    for ti, yi in zip(t, y):
        dt, _ = ti.astimezone_and_leap_second(tz)
        x = riseset[yi]
        year = str(dt.year)

        if year not in result.keys():
            result[year] = {'rises': [],
                            'rise_hr': [],
                            'sets': [],
                            'dates': [],
                            'set_hr': [],
                            'sunlight': [],
                            'solarnoons': [],
                            'betweensolarnoons': [],
                            }
        hrs = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        result[year][f"{x}s"].append(dt)
        result[year][f"{x}_hr"].append(hrs)
        if yi:
            prevrise = ti
            prevsolarnoon = solarnoon
            result[year]['dates'].append(dt.strftime('%Y-%m-%d'))
        else:
            solarnoon = solarnoons.pop(0)
            dtsn, _ = solarnoon.astimezone_and_leap_second(tz)
            result[year]['solarnoons'].append(dtsn)

            if prevsolarnoon is not None:
                hrs = (solarnoon - prevsolarnoon) * 24.0
                result[year]['betweensolarnoons'].append(hrs)
            if prevrise is not None:
                hrs = (ti - prevrise) * 24.0
                result[year]['sunlight'].append(hrs)

    return result


def risesetextremes(lat, lon, tzstr, startYear=2024, years=1, verbose=False):
    '''

    wrapper for

    :param lat:
    :param lon:
    :param tzstr:
    :param startYear:
    :param years:
    :param verbose:
    :return:
    '''
    lat2 = round(float(lat), 2)
    lon2 = round(float(lon), 2)
    if lat2 < 0:
        lat3 = str(abs(lat2)) + ' S'
    else:
        lat3 = str(abs(lat2)) + ' N'
    if lon2 < 0:
        lon3 = str(abs(lon2)) + ' W'
    else:
        lon3 = str(abs(lon2)) + ' E'
    if verbose: print('calculating sunrise, sunset')
    data = daily_solar_metrics(lat3, lon3, tzstr, startYear, years, verbose)
    if verbose: print('done')

    result = dict()

    for y in range(startYear, startYear + years):
        # Sun Distance
        result[str(y)] = equilux(lat, lon, tzstr, year=y, years=years)
        perihelion, aphelion = sundistance(y, tzstr)
        result[str(y)]['perihelion'] = perihelion[0]
        result[str(y)]['perihelion distance'] = round(perihelion[1] * 0.621371)
        result[str(y)]['aphelion'] = aphelion[0]
        result[str(y)]['aphelion distance'] = round(aphelion[1] * 0.621371)

        # earliest/latest sunset
        for x in ['rise', 'set']:
            result[str(y)][f"{x}_earliest"] = data[str(y)][f"{x}s"][
                data[str(y)][f"{x}_hr"].index(min(data[str(y)][f"{x}_hr"]))]
            result[str(y)][f"{x}_latest"] = data[str(y)][f"{x}s"][
                data[str(y)][f"{x}_hr"].index(max(data[str(y)][f"{x}_hr"]))]

        minmeridianday_index = data[str(y)][f"betweensolarnoons"].index(min(data[str(y)][f"betweensolarnoons"]))
        maxmeridianday_index = data[str(y)][f"betweensolarnoons"].index(max(data[str(y)][f"betweensolarnoons"]))
        result[str(y)][f"min_meridian_day"] = data[str(y)][f"solarnoons"][minmeridianday_index - 1]
        result[str(y)][f"max_meridian_day"] = data[str(y)][f"solarnoons"][maxmeridianday_index - 1]
        result[str(y)][f"min_meridian_day_hrs"] = data[str(y)][f"betweensolarnoons"][minmeridianday_index]
        result[str(y)][f"max_meridian_day_hrs"] = data[str(y)][f"betweensolarnoons"][maxmeridianday_index]

        # time above the horizon
        for x in ['short', 'long']:
            if x == 'short':
                f = min
            else:
                f = max
            idx = data[str(y)]["sunlight"].index(f(data[str(y)]["sunlight"]))
            hrs_fract, hrs = modf(data[str(y)]["sunlight"][idx])
            mins_fract, mins = modf(hrs_fract * 60.0)
            result[str(y)][f"{x}est_hrs"] = f"{int(hrs)} hours {int(mins)} minutes {round(mins_fract * 60.0)} seconds"
            result[str(y)][f"{x}est_date"] = data[str(y)][f"rises"][idx]
    return result, data
