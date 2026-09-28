create extension if not exists vector;

create table if not exists notes (
    id serial primary key,
    slug text not null,
    lang text not null default 'es',
    kind text not null,
    title text not null,
    tags text[] not null default '{}',
    summary text,
    link text,
    published date
);

-- Databases created before notes had a date or the 'note' kind.
alter table notes add column if not exists published date;
alter table notes drop constraint if exists notes_kind_check;
alter table notes add constraint notes_kind_check check (kind in ('project', 'skill', 'note', 'experience'));

-- Every note exists once per language: the same slug in 'es' and 'en'.
alter table notes add column if not exists lang text not null default 'es';
alter table notes drop constraint if exists notes_slug_key;
create unique index if not exists notes_slug_lang_idx on notes (slug, lang);

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

create table if not exists intents (
    id text not null,
    lang text not null default 'es',
    position int not null,
    label text not null,
    description text not null default '',
    note_slugs text[] not null default '{}'
);

alter table intents add column if not exists lang text not null default 'es';
alter table intents drop constraint if exists intents_pkey;
create unique index if not exists intents_id_lang_idx on intents (id, lang);

create table if not exists sessions (
    id uuid primary key,
    context vector({dim}),
    intent_belief double precision[],
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists events (
    id bigserial primary key,
    session_id uuid not null references sessions (id) on delete cascade,
    kind text not null check (kind in ('query', 'read', 'select', 'finish')),
    chunk_id text,
    note_id int references notes (id) on delete set null,
    query text,
    created_at timestamptz not null default now()
);

create index if not exists events_session_idx on events (session_id, created_at);
