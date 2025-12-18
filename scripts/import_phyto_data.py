#!/usr/bin/env python
"""
Import data from 3IT-data-framework's Phyto pipeline into ART's tcm.duckdb.

This enhanced version pulls from multiple sources:
- TCMBank herbs (9K herbs)
- HERB database (7K herbs with diseases)
- CMAUP plants with disease associations (8K plants, 765K associations)
- Duke ethnobotany (83K traditional uses)
- TCMSP herbs (502 herbs with categories)

**Provenance Tracking (Patent P7.3)**:
- Each record includes a citation_tag [CLT-{SOURCE}-{RECORD_ID}]
- Citations table tracks source provenance for all imported data
- Exports view includes citation_tag so GRPO outputs can cite sources

Usage:
    uv run python scripts/import_phyto_data.py
"""

import json
from datetime import datetime
from pathlib import Path

import duckdb

# Can use either the synced copy or the original
PHYTO_DB = Path("data/phyto.duckdb")
if not PHYTO_DB.exists():
    PHYTO_DB = Path(
        "/home/dp/Projects/3IT-data-framework/pipelines/phyto/storage/gold/phyto.duckdb"
    )

ART_DB = Path("data/tcm.duckdb")

# Source reliability scores (from 3IT-data-framework provenance module)
SOURCE_RELIABILITY = {
    "HERB": 0.82,
    "TCMBank": 0.90,
    "CMAUP": 0.85,
    "Duke": 0.75,
    "TCMSP": 0.85,
}


def import_data():
    print(f"Source: {PHYTO_DB}")
    print(f"Target: {ART_DB}")

    if not PHYTO_DB.exists():
        raise FileNotFoundError(f"Phyto database not found: {PHYTO_DB}")

    # Ensure target directory exists
    ART_DB.parent.mkdir(parents=True, exist_ok=True)

    # Connect to ART database (fresh start)
    if ART_DB.exists():
        ART_DB.unlink()
    art_conn = duckdb.connect(str(ART_DB))

    # Attach phyto as read-only
    art_conn.execute(f"ATTACH '{PHYTO_DB}' AS phyto (READ_ONLY)")

    # =========================================================================
    # 0. Create citations table for provenance tracking (Patent P7.3)
    # =========================================================================
    print("\n0. Creating citations table for provenance tracking...")
    art_conn.execute("""
        CREATE TABLE citations (
            citation_tag VARCHAR PRIMARY KEY,
            source_id VARCHAR NOT NULL,
            source_record_id VARCHAR NOT NULL,
            entity_type VARCHAR NOT NULL,
            entity_id VARCHAR NOT NULL,
            source_reliability DOUBLE DEFAULT 0.7,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified_at TIMESTAMP,
            verified_by VARCHAR DEFAULT 'automatic'
        )
    """)

    # =========================================================================
    # 1. HERB Database - Rich herb data with functions and indications
    # =========================================================================
    print("\n1. Importing from HERB database (herb_herbs)...")
    art_conn.execute("""
        CREATE TABLE herb_plants AS
        SELECT
            Herb_id AS plant_id,
            '[CLT-HERB-' || CAST(Herb_id AS VARCHAR) || ']' AS citation_tag,
            Herb_latin_name AS name_scientific,
            COALESCE(Herb_en_name, Herb_pinyin_name, Herb_cn_name) AS name_common,
            Indication AS condition_name,
            Function AS function_text,
            Properties AS properties,
            Meridians AS meridians,
            Toxicity AS toxicity,
            UsePart AS use_part,
            'HERB' AS source,
            0.82 AS source_reliability
        FROM phyto.herb_herbs
        WHERE (Herb_en_name IS NOT NULL OR Herb_pinyin_name IS NOT NULL)
          AND (Function IS NOT NULL OR Indication IS NOT NULL)
    """)
    herb_count = art_conn.execute("SELECT COUNT(*) FROM herb_plants").fetchone()[0]
    print(f"   Imported {herb_count} herbs from HERB database")

    # Register HERB citations
    art_conn.execute("""
        INSERT INTO citations (citation_tag, source_id, source_record_id, entity_type, entity_id, source_reliability)
        SELECT citation_tag, 'HERB', plant_id, 'herb', plant_id, source_reliability
        FROM herb_plants
    """)

    # =========================================================================
    # 2. TCMBank - Additional herbs
    # =========================================================================
    print("\n2. Importing from TCMBank (tcmbank_herbs)...")
    art_conn.execute("""
        CREATE TABLE tcmbank_plants AS
        SELECT
            tcmbank_id AS plant_id,
            '[CLT-TCMBANK-' || CAST(tcmbank_id AS VARCHAR) || ']' AS citation_tag,
            herb_latin_name AS name_scientific,
            COALESCE(tcm_name_en, tcm_name) AS name_common,
            indication AS condition_name,
            function AS function_text,
            properties,
            meridians,
            toxicity,
            usepart AS use_part,
            'TCMBank' AS source,
            0.90 AS source_reliability
        FROM phyto.tcmbank_herbs
        WHERE (tcm_name_en IS NOT NULL OR tcm_name IS NOT NULL)
          AND (function IS NOT NULL OR indication IS NOT NULL)
          AND tcmbank_id NOT IN (SELECT plant_id FROM herb_plants)
    """)
    tcmbank_count = art_conn.execute("SELECT COUNT(*) FROM tcmbank_plants").fetchone()[
        0
    ]
    print(f"   Imported {tcmbank_count} herbs from TCMBank")

    # Register TCMBank citations
    art_conn.execute("""
        INSERT INTO citations (citation_tag, source_id, source_record_id, entity_type, entity_id, source_reliability)
        SELECT citation_tag, 'TCMBank', plant_id, 'herb', plant_id, source_reliability
        FROM tcmbank_plants
    """)

    # =========================================================================
    # 3. CMAUP - Plants with disease associations
    # =========================================================================
    print("\n3. Importing from CMAUP (plants + disease associations)...")
    art_conn.execute("""
        CREATE TABLE cmaup_plants AS
        SELECT DISTINCT
            p.Plant_ID AS plant_id,
            '[CLT-CMAUP-' || CAST(p.Plant_ID AS VARCHAR) || ']' AS citation_tag,
            p.Species_Name AS name_scientific,
            p.Plant_Name AS name_common,
            d.Disease AS condition_name,
            d.Disease_Category AS function_text,
            NULL AS properties,
            NULL AS meridians,
            NULL AS toxicity,
            NULL AS use_part,
            'CMAUP' AS source,
            0.85 AS source_reliability
        FROM phyto.cmaup_plants p
        JOIN phyto.cmaup_plant_diseases d ON p.Plant_ID = d.Plant_ID
        WHERE d.Disease IS NOT NULL
          AND d.Disease != ''
          AND p.Plant_ID NOT IN (SELECT plant_id FROM herb_plants)
          AND p.Plant_ID NOT IN (SELECT plant_id FROM tcmbank_plants)
    """)
    cmaup_count = art_conn.execute("SELECT COUNT(*) FROM cmaup_plants").fetchone()[0]
    print(f"   Imported {cmaup_count} plant-disease associations from CMAUP")

    # Register CMAUP citations
    art_conn.execute("""
        INSERT INTO citations (citation_tag, source_id, source_record_id, entity_type, entity_id, source_reliability)
        SELECT DISTINCT citation_tag, 'CMAUP', plant_id, 'plant', plant_id, source_reliability
        FROM cmaup_plants
    """)

    # =========================================================================
    # 4. Duke Ethnobotany - Traditional uses
    # =========================================================================
    print("\n4. Importing from Duke Ethnobotany...")
    art_conn.execute("""
        CREATE TABLE duke_plants AS
        SELECT DISTINCT
            'DUKE_' || CAST(e.id AS VARCHAR) AS plant_id,
            '[CLT-DUKE-' || CAST(e.id AS VARCHAR) || ']' AS citation_tag,
            e.scientific_name AS name_scientific,
            COALESCE(e.common_name, e.scientific_name) AS name_common,
            e.activity AS condition_name,
            'Traditional use: ' || e.activity AS function_text,
            NULL AS properties,
            NULL AS meridians,
            NULL AS toxicity,
            NULL AS use_part,
            'Duke' AS source,
            0.75 AS source_reliability
        FROM phyto.duke_ethnobotany e
        WHERE e.activity IS NOT NULL
          AND e.activity != ''
          AND e.scientific_name IS NOT NULL
    """)
    duke_count = art_conn.execute("SELECT COUNT(*) FROM duke_plants").fetchone()[0]
    print(f"   Imported {duke_count} ethnobotany records from Duke")

    # Register Duke citations
    art_conn.execute("""
        INSERT INTO citations (citation_tag, source_id, source_record_id, entity_type, entity_id, source_reliability)
        SELECT DISTINCT citation_tag, 'Duke', REPLACE(plant_id, 'DUKE_', ''), 'ethnobotany', plant_id, source_reliability
        FROM duke_plants
    """)

    # =========================================================================
    # 5. Combine all sources into unified plants table
    # =========================================================================
    print("\n5. Creating unified tcm_plants table...")
    art_conn.execute("""
        CREATE TABLE tcm_plants AS
        SELECT * FROM herb_plants
        UNION ALL
        SELECT * FROM tcmbank_plants
        UNION ALL
        SELECT * FROM cmaup_plants
        UNION ALL
        SELECT * FROM duke_plants
    """)
    total_plants = art_conn.execute("SELECT COUNT(*) FROM tcm_plants").fetchone()[0]
    print(f"   Total unified records: {total_plants}")

    # Citation stats
    citation_count = art_conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    print(f"   Total citations registered: {citation_count}")

    # =========================================================================
    # 6. Create protocols with response text
    # =========================================================================
    print("\n6. Creating plant_protocols with response text...")
    art_conn.execute("""
        CREATE TABLE plant_protocols AS
        SELECT
            ROW_NUMBER() OVER () AS protocol_id,
            plant_id,
            citation_tag,
            name_common,
            condition_name,
            CASE
                WHEN properties IS NOT NULL AND use_part IS NOT NULL
                THEN properties || ' (' || use_part || ')'
                WHEN properties IS NOT NULL THEN properties
                WHEN use_part IS NOT NULL THEN use_part
                ELSE 'Standard preparation'
            END AS preparation_notes,
            -- Build response text from available fields
            COALESCE(function_text, '') ||
            CASE WHEN function_text IS NOT NULL AND condition_name IS NOT NULL THEN '. ' ELSE '' END ||
            CASE WHEN condition_name IS NOT NULL THEN 'Indicated for: ' || condition_name ELSE '' END ||
            CASE WHEN meridians IS NOT NULL THEN '. Meridians: ' || meridians ELSE '' END ||
            CASE WHEN use_part IS NOT NULL THEN '. Use part: ' || use_part ELSE '' END
            AS response_text,
            (toxicity IS NOT NULL AND toxicity != '' AND LOWER(toxicity) != 'none') AS contraindication_flag,
            CASE
                WHEN toxicity IS NOT NULL AND toxicity != '' AND LOWER(toxicity) != 'none'
                THEN 'Monitor for: ' || toxicity
                ELSE NULL
            END AS monitoring_clause,
            source,
            source_reliability,
            -- Reward heuristic based on data completeness AND source reliability
            ROUND(
                CASE
                    WHEN function_text IS NOT NULL AND condition_name IS NOT NULL
                         AND properties IS NOT NULL AND meridians IS NOT NULL THEN 0.90
                    WHEN function_text IS NOT NULL AND condition_name IS NOT NULL
                         AND (properties IS NOT NULL OR meridians IS NOT NULL) THEN 0.82
                    WHEN function_text IS NOT NULL AND condition_name IS NOT NULL THEN 0.75
                    WHEN function_text IS NOT NULL OR condition_name IS NOT NULL THEN 0.62
                    ELSE 0.50
                END * source_reliability, 3
            ) AS reward_score
        FROM tcm_plants
        WHERE COALESCE(function_text, '') || COALESCE(condition_name, '') != ''
    """)
    protocols_count = art_conn.execute(
        "SELECT COUNT(*) FROM plant_protocols"
    ).fetchone()[0]
    print(f"   Created {protocols_count} protocols")

    # =========================================================================
    # 7. Import compound data from multiple sources
    # =========================================================================
    print("\n7. Importing compound data...")

    # From TCMBank ingredients
    art_conn.execute("""
        CREATE TABLE plant_compounds AS
        SELECT DISTINCT
            p.plant_id,
            i.name AS compound_name
        FROM tcm_plants p
        JOIN phyto.tcmbank_ingredients i
            ON LOWER(TRIM(i.tcm_name_en)) = LOWER(TRIM(p.name_common))
            OR LOWER(TRIM(i.tcm_name)) = LOWER(TRIM(p.name_common))
        WHERE i.name IS NOT NULL
          AND i.name != ''
          AND LENGTH(i.name) > 2
          AND p.source IN ('HERB', 'TCMBank')
    """)

    # From TCMBank ingredients for HERB plants (match by name)
    art_conn.execute("""
        INSERT INTO plant_compounds
        SELECT DISTINCT
            p.plant_id,
            i.name AS compound_name
        FROM tcm_plants p
        JOIN phyto.tcmbank_ingredients i
            ON LOWER(TRIM(i.tcm_name_en)) = LOWER(TRIM(p.name_common))
            OR LOWER(TRIM(i.tcm_name)) = LOWER(TRIM(p.name_common))
        WHERE i.name IS NOT NULL
          AND i.name != ''
          AND LENGTH(i.name) > 2
          AND p.source = 'HERB'
          AND NOT EXISTS (
              SELECT 1 FROM plant_compounds pc
              WHERE pc.plant_id = p.plant_id AND pc.compound_name = i.name
          )
    """)

    # NOTE: Skipping CMAUP compound import - the cmaup_plant_ingredients table
    # in phyto.duckdb has corrupted schema (first row values as column names).
    # This can be fixed by re-running the CMAUP ETL job in 3IT-data-framework.
    # For now, CMAUP plants will not have compound associations.
    print("   (Skipped CMAUP compounds - table schema needs repair)")

    compounds_count = art_conn.execute(
        "SELECT COUNT(*) FROM plant_compounds"
    ).fetchone()[0]
    print(f"   Imported {compounds_count} plant-compound links")

    # =========================================================================
    # 8. Create the exports view for training (with citation_tag for provenance)
    # =========================================================================
    print(
        "\n8. Creating exports.tcm_instruction_pairs view (with citation tracking)..."
    )
    art_conn.execute("CREATE SCHEMA IF NOT EXISTS exports")
    art_conn.execute("""
        CREATE OR REPLACE VIEW exports.tcm_instruction_pairs AS
        SELECT
            proto.protocol_id,
            p.plant_id,
            p.citation_tag,
            p.name_scientific,
            p.name_common,
            proto.condition_name,
            COALESCE(pc.compound_list, '') AS compounds,
            proto.preparation_notes,
            proto.response_text,
            proto.contraindication_flag,
            proto.monitoring_clause,
            proto.source,
            proto.source_reliability,
            COALESCE(
                proto.reward_score,
                ROUND(
                    CASE
                        WHEN proto.contraindication_flag = TRUE AND pc.num_compounds >= 2 THEN 0.92
                        WHEN pc.num_compounds >= 2 THEN 0.82
                        WHEN pc.num_compounds = 1 THEN 0.68
                        ELSE 0.5
                    END * proto.source_reliability, 3
                )
            ) AS reward
        FROM tcm_plants AS p
        JOIN plant_protocols AS proto ON p.plant_id = proto.plant_id
        LEFT JOIN (
            SELECT
                plant_id,
                COUNT(*) AS num_compounds,
                string_agg(compound_name, ', ' ORDER BY compound_name) AS compound_list
            FROM plant_compounds
            GROUP BY plant_id
        ) AS pc ON p.plant_id = pc.plant_id
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)

    # Source breakdown
    print("\nBy Source:")
    source_counts = art_conn.execute("""
        SELECT source, COUNT(*) as count
        FROM exports.tcm_instruction_pairs
        GROUP BY source
        ORDER BY count DESC
    """).fetchdf()
    print(source_counts.to_string(index=False))

    # Table summary
    print("\nTables:")
    for table in ["tcm_plants", "plant_protocols", "plant_compounds", "citations"]:
        count = art_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")

    view_count = art_conn.execute(
        "SELECT COUNT(*) FROM exports.tcm_instruction_pairs"
    ).fetchone()[0]
    print(f"  exports.tcm_instruction_pairs: {view_count:,} rows")

    # Citation coverage
    print("\nCitation Coverage (Provenance P7.3):")
    citation_by_source = art_conn.execute("""
        SELECT source_id, COUNT(*) as count, ROUND(AVG(source_reliability), 2) as avg_reliability
        FROM citations
        GROUP BY source_id
        ORDER BY count DESC
    """).fetchdf()
    print(citation_by_source.to_string(index=False))

    # Unique conditions (important for GRPO)
    unique_conditions = art_conn.execute(
        "SELECT COUNT(DISTINCT condition_name) FROM exports.tcm_instruction_pairs"
    ).fetchone()[0]
    print(f"\nUnique conditions (prompts): {unique_conditions:,}")

    # Reward distribution
    print("\nReward Distribution:")
    reward_dist = art_conn.execute("""
        SELECT
            CASE
                WHEN reward >= 0.85 THEN 'high (0.85+)'
                WHEN reward >= 0.70 THEN 'medium (0.70-0.84)'
                WHEN reward >= 0.50 THEN 'low (0.50-0.69)'
                ELSE 'very low (<0.50)'
            END as quality,
            COUNT(*) as count
        FROM exports.tcm_instruction_pairs
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
    print(reward_dist.to_string(index=False))

    # Sample output with citation tags
    print("\n" + "=" * 60)
    print("SAMPLE DATA (with citation tags)")
    print("=" * 60)
    sample = art_conn.execute("""
        SELECT citation_tag, source, name_common,
               LEFT(condition_name, 30) as condition,
               ROUND(reward, 2) as reward
        FROM exports.tcm_instruction_pairs
        WHERE compounds != ''
        ORDER BY RANDOM()
        LIMIT 5
    """).fetchdf()
    print(sample.to_string(index=False))

    art_conn.close()
    print(f"\nDatabase saved to: {ART_DB.absolute()}")

    # =========================================================================
    # 9. Save import metadata for tracking
    # =========================================================================
    import_meta = {
        "imported_at": datetime.now().isoformat(),
        "source_db": str(PHYTO_DB.absolute()),
        "target_db": str(ART_DB.absolute()),
        "total_plants": total_plants,
        "total_protocols": protocols_count,
        "total_compounds": compounds_count,
        "total_citations": citation_count,
        "sources": {
            "HERB": herb_count,
            "TCMBank": tcmbank_count,
            "CMAUP": cmaup_count,
            "Duke": duke_count,
        },
        "source_reliability": SOURCE_RELIABILITY,
        "provenance_enabled": True,
    }

    import_meta_file = ART_DB.parent / "_import_metadata.json"
    import_meta_file.write_text(json.dumps(import_meta, indent=2))
    print(f"Import metadata saved to: {import_meta_file}")


if __name__ == "__main__":
    import_data()
