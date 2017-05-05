drop table if exists entries;
create table entries (
  id integer primary key autoincrement,
  resident_id text not null,
  address text not null,
  lon text not null,
  lat text not null,
  curve_id num not null
);




