# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:40:51 2017

@author: Jamie
"""
from __future__ import print_function # In python 2.7
from Model3 import cluster_curves, k_means, average_day, scale_data, return_curve
import pandas as pd
import numpy as np
import os
import random
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
import json

app = Flask(__name__)

app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'flaskr.db'),
    #DEBUG=True,
    #SECRET_KEY='development key',
    #USERNAME='admin',
    #PASSWORD='default'
))
def connect_db():
    """Connects to the specific database."""
    rv = sqlite3.connect(app.config['DATABASE'])
    rv.row_factory = sqlite3.Row
    return rv


def init_db():
    """Initializes the database."""
    db = get_db()
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()


@app.cli.command('initdb')
def initdb_command():
    """Creates the database tables."""
    init_db()
    print('Initialized the database.')


def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

@app.route("/")
def main():
    return render_template('index.html')

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route("/showEntries", methods=['GET', 'POST'])
def showEntries():
    select = request.form.get('prof_select')
    db = get_db()
    #cur = db.execute('select * from entries order by id desc')
    if select is None:
        cur = db.execute('select * from entries order by id desc')
    else:
        cur = db.execute('select * from entries where curve_id = ?', (select,))
    
    entries = cur.fetchall()
    markers = []
    for e in entries:
        markers.append(Marker(e[3], e[4], e[1], e[2]))
    return render_template('show_entries.html', entries=entries, 
                           data=[{'name':'All'}, {'name':'1'}, {'name':'2'}, 
                                 {'name':'3'}, {'name':'4'}],
                           markers=json.dumps([m.__dict__ for m in markers]))
    
class Marker:
    def __init__(self, lon, lat, rid, address):
        self.lon = lon
        self.lat = lat
        self.rid = rid
        self.address = address
   
@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.get('prof_select')
    return(str(select)) # just to see what select is
   
@app.route("/initializeDB")
def initializeDB():
    db = get_db()
    db.execute('DELETE FROM entries')
    #entrie - #location - #
    #df = pd.read_csv('/Users/user/python-introduction-jhussman/FinalProject/compiled_data.csv')
    #df = df.drop(df.index[len(df)-1])
    #df = average_day(df)
    #df = scale_data(df)
    locations = pd.read_csv('//Users/user/python-introduction-jhussman/FinalProject/Data/cleaned_locations.csv',converters={'resident_id': str,'LON': str,'LAT': str,'ADDRRESS': str})
    resident_id = list(locations.resident_id.values)
    lon = list(locations.LON.values)
    lat = list(locations.LAT.values)
    #curve_id = list(locations.curve_id.values)
    address = list(locations.ADDRESS.values)
    #curve_id =[random.randrange(0,3,1) for _ in range (363)]
    curve_id = bla
    #curve_id = curve_id_plot(df,3)
    for item,item2,item3,item4,item5 in zip(resident_id,address,lon,lat,curve_id):
        db.execute('insert into entries values (?,?,?,?,?,?)', (None,item,item2,item3,item4,item5))
        db.commit() 
    return render_template('index.html')  
    
@app.route("/showLoadCurves")
def showLoadCurves():
    return render_template('show_load_curves.html')  
    
    
@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return

if __name__ == "__main__":
    app.run()
    
