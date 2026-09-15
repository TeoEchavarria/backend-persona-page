create extension if not exists vector;

create table if not exists notes (
    id serial primary key,
    slug text unique not null,
    kind text not null check (kind in ('project', 'skill')),
    title text not null,
    tags text[] not null default '{}',
    summary text,
    link text
);

create table if not exists chunks (
    id text primary key,
    note_id int not null references notes (id) on delete cascade,
    position int not null,
    headings text[] not null default '{}',
    text text not null,
    text_with_context text not null,
    embedding vector({dim}) not null,
    tsv tsvector generated always as (to_tsvector('{text_config}', text_with_context)) stored
);

create index if not exists chunks_tsv_idx on chunks using gin (tsv);
create index if not exists chunks_note_idx on chunks (note_id, position);
