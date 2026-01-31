"""Read-only access to Zotero SQLite database."""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paper:
    """Represents a paper from the Zotero library."""

    key: str
    title: str
    year: int | None
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pdf_path: Path | None = None


def get_papers(
    db_path: Path,
    storage_path: Path,
    tags: list[str] | None = None,
    collections: list[str] | None = None,
) -> list[Paper]:
    """Extract papers from Zotero database with optional filtering."""
    # Use immutable mode to read even when Zotero is running
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row

    try:
        item_ids = _get_filtered_item_ids(conn, tags, collections)
        if not item_ids:
            return []

        papers = []
        for item_id in item_ids:
            paper = _build_paper(conn, item_id, storage_path)
            if paper:
                papers.append(paper)
        return papers
    finally:
        conn.close()


def _get_filtered_item_ids(
    conn: sqlite3.Connection,
    tags: list[str] | None,
    collections: list[str] | None,
) -> list[int]:
    """Get item IDs matching tag/collection filters."""
    # Get all top-level items (not attachments/notes)
    query = """
        SELECT items.itemID
        FROM items
        JOIN itemTypes ON items.itemTypeID = itemTypes.itemTypeID
        WHERE itemTypes.typeName NOT IN ('attachment', 'note', 'annotation')
        AND items.itemID NOT IN (SELECT itemID FROM deletedItems)
    """
    cursor = conn.execute(query)
    all_ids = {row["itemID"] for row in cursor}

    if tags:
        tag_ids = _get_ids_by_tags(conn, tags)
        all_ids &= tag_ids

    if collections:
        collection_ids = _get_ids_by_collections(conn, collections)
        all_ids &= collection_ids

    return list(all_ids)


def _get_ids_by_tags(conn: sqlite3.Connection, tags: list[str]) -> set[int]:
    """Get item IDs that have any of the specified tags."""
    placeholders = ",".join("?" * len(tags))
    query = f"""
        SELECT DISTINCT itemTags.itemID
        FROM itemTags
        JOIN tags ON itemTags.tagID = tags.tagID
        WHERE tags.name IN ({placeholders})
    """
    cursor = conn.execute(query, tags)
    return {row["itemID"] for row in cursor}


def _get_ids_by_collections(conn: sqlite3.Connection, collections: list[str]) -> set[int]:
    """Get item IDs that are in any of the specified collections."""
    placeholders = ",".join("?" * len(collections))
    query = f"""
        SELECT DISTINCT collectionItems.itemID
        FROM collectionItems
        JOIN collections ON collectionItems.collectionID = collections.collectionID
        WHERE collections.collectionName IN ({placeholders})
    """
    cursor = conn.execute(query, collections)
    return {row["itemID"] for row in cursor}


def _build_paper(conn: sqlite3.Connection, item_id: int, storage_path: Path) -> Paper | None:
    """Build a Paper object from database rows."""
    # Get item key
    cursor = conn.execute("SELECT key FROM items WHERE itemID = ?", (item_id,))
    row = cursor.fetchone()
    if not row:
        return None
    key = row["key"]

    # Get metadata fields
    metadata = _get_item_metadata(conn, item_id)
    title = metadata.get("title", "")
    if not title:
        return None  # Skip items without titles

    year = None
    date_str = metadata.get("date", "")
    if date_str:
        # Extract year from various date formats (e.g., "2023", "2023-01-15")
        year_part = date_str.split("-")[0].strip()
        if year_part.isdigit() and len(year_part) == 4:
            year = int(year_part)

    return Paper(
        key=key,
        title=title,
        year=year,
        authors=_get_authors(conn, item_id),
        abstract=metadata.get("abstractNote", ""),
        tags=_get_tags(conn, item_id),
        notes=_get_notes(conn, item_id),
        pdf_path=_get_pdf_path(conn, item_id, storage_path),
    )


def _get_item_metadata(conn: sqlite3.Connection, item_id: int) -> dict[str, str]:
    """Get metadata fields for an item."""
    query = """
        SELECT fields.fieldName, itemDataValues.value
        FROM itemData
        JOIN fields ON itemData.fieldID = fields.fieldID
        JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
        WHERE itemData.itemID = ?
    """
    cursor = conn.execute(query, (item_id,))
    return {row["fieldName"]: row["value"] for row in cursor}


def _get_authors(conn: sqlite3.Connection, item_id: int) -> list[str]:
    """Get author names for an item."""
    query = """
        SELECT creators.firstName, creators.lastName
        FROM itemCreators
        JOIN creators ON itemCreators.creatorID = creators.creatorID
        WHERE itemCreators.itemID = ?
        ORDER BY itemCreators.orderIndex
    """
    cursor = conn.execute(query, (item_id,))
    authors = []
    for row in cursor:
        first = row["firstName"] or ""
        last = row["lastName"] or ""
        name = f"{first} {last}".strip() if first else last
        if name:
            authors.append(name)
    return authors


def _get_tags(conn: sqlite3.Connection, item_id: int) -> list[str]:
    """Get tags for an item."""
    query = """
        SELECT tags.name
        FROM itemTags
        JOIN tags ON itemTags.tagID = tags.tagID
        WHERE itemTags.itemID = ?
    """
    cursor = conn.execute(query, (item_id,))
    return [row["name"] for row in cursor]


def _get_notes(conn: sqlite3.Connection, item_id: int) -> list[str]:
    """Get notes attached to an item."""
    query = """
        SELECT itemNotes.note
        FROM itemNotes
        JOIN items ON itemNotes.itemID = items.itemID
        WHERE itemNotes.parentItemID = ?
    """
    cursor = conn.execute(query, (item_id,))
    return [row["note"] for row in cursor if row["note"]]


def _get_pdf_path(conn: sqlite3.Connection, item_id: int, storage_path: Path) -> Path | None:
    """Get the path to the first PDF attachment."""
    query = """
        SELECT items.key, itemAttachments.path
        FROM itemAttachments
        JOIN items ON itemAttachments.itemID = items.itemID
        WHERE itemAttachments.parentItemID = ?
        AND itemAttachments.contentType = 'application/pdf'
    """
    cursor = conn.execute(query, (item_id,))
    row = cursor.fetchone()
    if not row:
        return None

    attachment_key = row["key"]
    path_value = row["path"]

    if path_value and path_value.startswith("storage:"):
        filename = path_value[8:]  # Remove "storage:" prefix
        pdf_path = storage_path / attachment_key / filename
        if pdf_path.exists():
            return pdf_path

    return None
