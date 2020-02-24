#!/usr/bin/env python

import sqlite3
import pandas as pd

# create db connection
conn = sqlite3.connect("sampleDB.db")

try:
    conn.execute("""
    PRAGMA foreign_keys = ON; 
    """)
    
    #instantiate tables for sample db
    conn.execute('''
    CREATE TABLE sample (id INT, date TEXT, sequencer_id TEXT, tissue TEXT, species TEXT, seed_id INT, collection_location TEXT, experiment TEXT, data_size INT, PRIMARY KEY ('id'),
        FOREIGN KEY ('seed_id')
        REFERENCES genotype ('seed_id')
        FOREIGN KEY ('collection_location')
        REFERENCES location ('location_id')
        FOREIGN KEY ('sequencer_id')
        REFERENCES sequencer ('machine_id')
        FOREIGN KEY ('experiment')
        REFERENCES experiment ('experiment_name'));
    ''')
    conn.execute('''
    CREATE TABLE genotype (seed_id INT, species TEXT, genotype TEXT, seeds_left INT, PRIMARY KEY ('seed_id')); 
    ''')
    conn.execute('''
    CREATE TABLE location (location_id TEXT, longitude REAL, latitude REAL, PRIMARY KEY ('location_id'));
    ''')
    conn.execute('''
    CREATE TABLE sequencer (machine_id TEXT, type TEXT, year INT, PRIMARY KEY ('machine_id'));
    ''')   
    conn.execute('''
    CREATE TABLE experiment (experiment_name TEXT, grant_id INT, cost REAL, PRIMARY KEY ('experiment_name'), 
        FOREIGN KEY ('grant_id')
        REFERENCES grant ('grant_id'));
    ''')
    conn.execute('''
    CREATE TABLE grant (grant_id INT, pi TEXT, funding_agency TEXT, PRIMARY KEY ('grant_id')); 
    ''')
    
    #add values to tables
    conn.executemany("""
    INSERT INTO genotype VALUES (?,?,?,?)""", 
                     [(1, 'oryza sativa', 'os702', 50),
                      (2, 'zea mays', 'b73', 70),
                      (3, 'oryza sativa', 'os-sw', 3),
                      (4, 'panicum virgatum', 'green90', 22),
                      (5, 'arabidopsis thaliana', 'at23', 150),
                      (10, 'utricularia gibba', 'ugibba-wl', 203),
                     ]
    )
    conn.executemany("""
    INSERT INTO location VALUES (?,?,?)""", 
                     [('iron_horse', 38.423, 37.28),
                      ('riverbend_gh', 38.427, 37.27),
                     ]
    )
    conn.executemany("""
    INSERT INTO sequencer VALUES (?,?,?)""", 
                     [('20s5j', 'illumina_miseq', 2015),
                      ('3a79y', 'pacbio', 2018),
                     ]
    )
    conn.executemany("""
    INSERT INTO grant VALUES (?,?,?)""", 
                     [(200978, 'mary_scientist', 'NSF'),
                      (90987, 'joe_pipette', 'NSF'),
                      (14745, 'ima_computer', 'UGA'),
                     ]
    )
    conn.executemany("""
    INSERT INTO experiment VALUES (?,?,?)""", 
                     [('grass_evolution', 200978, 50000.70),
                      ('bladderwort', 90987, 120050.00),
                      ('alignment_optimization', 14745, 15023.20),
                     ]
    )
    conn.executemany("""
    INSERT INTO sample VALUES (?,?,?,?,?,?,?,?,?)""", 
                     [(1, '2017-01-02', '20s5j', 'leaf', 'zea mays', 2, 'iron_horse', 'grass_evolution', 1923492), 
                      (2, '2017-03-29', '3a79y','leaf', 'utricularia gibba', 10, 'iron_horse', 'bladderwort', 123232),
                      (3, '2018-11-17', '20s5j','stem', 'zea mays', 2, 'iron_horse', 'grass_evolution', 897493),
                      (4, '2019-07-02', '20s5j','stem', 'oryza sativa', 1, 'iron_horse', 'grass_evolution', 1929309),
                      (5, '2019-07-10', '20s5j','root', 'oryza sativa', 3, 'iron_horse', 'grass_evolution', 1898332),
                     ]
    )
    conn.commit();
finally:
    print("this executed")
    conn.close()
