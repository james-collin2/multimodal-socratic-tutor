"""
scripts/ingest_corpus.py

Complete corpus ingestion pipeline for socratOT.
Reads PDF directly → chunks → embeds → indexes into ChromaDB.

Usage:
    python scripts/ingest_corpus.py              # full pipeline (PDF + sample)
    python scripts/ingest_corpus.py --sample     # built-in sample only (no internet)
    python scripts/ingest_corpus.py --pdf PATH   # specific PDF file
    python scripts/ingest_corpus.py --embed-only # re-embed existing chunks
    python scripts/ingest_corpus.py --reset      # clear store and re-index
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import logger

# ── Built-in sample corpus ────────────────────────────────────────────────────

SAMPLE_TEXTS = {
    "cerebellum": """
The Cerebellum and Motor Coordination in Occupational Therapy

The cerebellum is located at the posterior and inferior part of the brain,
beneath the occipital lobes of the cerebral cortex. It accounts for
approximately 10 percent of brain volume but contains nearly half of all
neurons in the entire brain.

The cerebellum coordinates voluntary movements, maintains balance and
equilibrium, and fine-tunes motor activity. It does not initiate movements
but refines them by comparing intended movements with actual movements.

Structure of the Cerebellum:
The cerebellum has two hemispheres separated by the vermis. The surface
is covered in folds called folia. The outer layer is the cerebellar cortex
containing three layers: molecular layer, Purkinje cell layer, granular layer.

Purkinje cells are the primary output neurons of the cerebellar cortex.
They send inhibitory signals to the deep cerebellar nuclei including the
dentate nucleus, interposed nuclei, and fastigial nucleus. The dentate
nucleus is the largest and projects to the thalamus then motor cortex.

Clinical relevance for Occupational Therapy:
Cerebellar dysfunction causes ataxia, dysmetria, intention tremor, and
dysdiadochokinesia. These directly affect ADLs including writing, eating,
dressing, and fine motor tasks. OT practitioners use graded motor activities,
adaptive equipment, task modification, and compensatory techniques.
Frenkel exercises use slow controlled repetitive movements for cerebellar ataxia.
""",
    "cranial_nerves": """
Cranial Nerves and Occupational Therapy Practice

Twelve pairs of cranial nerves emerge from the brain and brainstem,
numbered I through XII. Each has specific sensory, motor, or mixed functions.

CN V Trigeminal: three branches (V1 ophthalmic, V2 maxillary, V3 mandibular).
Provides sensation to face, oral mucosa, teeth, anterior tongue.
Motor: muscles of mastication. OT: affects eating and oral hygiene.

CN VII Facial: mixed nerve. Motor controls all facial expression muscles
including frontalis, orbicularis oculi, zygomaticus, orbicularis oris.
Sensory carries taste from anterior two-thirds of tongue via chorda tympani.
Facial nerve palsy: ipsilateral weakness of all facial muscles.
OT: affects eating, drinking, oral hygiene, social participation.
Treatment: facial exercises, neuromuscular electrical stimulation.

CN X Vagus: longest cranial nerve to thorax and abdomen.
Parasympathetic to thoracic and abdominal organs.
Motor to pharynx and larynx — critical for swallowing and speech.
OT: dysphagia management, respiratory function, autonomic regulation.

CN XI Accessory: motor to sternocleidomastoid and trapezius.
Critical for head movement and shoulder elevation.
OT: shoulder stability, overhead reaching, ADL tasks.

CN XII Hypoglossal: motor to all intrinsic and extrinsic tongue muscles.
Damage causes tongue deviation toward affected side and dysarthria.
OT: oral motor evaluation, eating, drinking, verbal communication in ADLs.

Assessment tools: Mann Assessment of Swallowing Ability (MASA),
Functional Oral Intake Scale (FOIS), Bedside Swallowing Assessment.
""",
    "brachial_plexus": """
Brachial Plexus and Upper Extremity Innervation

The brachial plexus is formed by ventral rami of C5 through T1.
It provides complete motor and sensory innervation to the upper extremity.

Organisation: Roots → Trunks → Divisions → Cords → Terminal branches
Upper trunk (C5-C6), Middle trunk (C7), Lower trunk (C8-T1)
Lateral cord, Medial cord, Posterior cord

Five terminal branches:
1. Musculocutaneous (C5-C7): biceps, brachialis, coracobrachialis
2. Median (C6-T1): flexor forearm, thenar muscles, lateral two lumbricals
   Sensory: lateral palm, thumb, index, middle, lateral ring finger
3. Ulnar (C8-T1): FCU, medial FDP, all intrinsics except thenar
   Sensory: medial palm, little and medial ring fingers
4. Radial (C5-T1): all extensor compartment muscles
   Sensory: posterior arm and forearm, dorsal radial hand
5. Axillary (C5-C6): deltoid, teres minor. Sensory: lateral shoulder

Median nerve conditions — Carpal tunnel syndrome:
Compression at carpal tunnel: thenar atrophy, weak pinch, median numbness.
OT: neutral wrist splinting, activity modification, ergonomic assessment,
nerve gliding exercises, sensory re-education.

Ulnar nerve conditions — Cubital tunnel syndrome:
Claw deformity ring and little fingers, weak grip, Froment sign positive.
OT: elbow extension splinting, anti-claw splinting, ADL modification.

Radial nerve — Saturday night palsy:
Wrist drop, cannot extend fingers. Dynamic wrist extension splint in OT.

Brachial plexus injuries:
Erb palsy (C5-C6): waiter tip position — shoulder adduction, elbow extension.
Klumpke palsy (C8-T1): intrinsic paralysis, claw hand, Horner syndrome.
OT: range of motion, splinting, electrical stimulation, sensory re-education.
""",
    "hand_anatomy": """
Hand Anatomy and Functional Significance in Occupational Therapy

The hand is the primary prehensile organ. Complex anatomy enables
grasps, pinches, and manipulation essential for occupational performance.

Bony architecture — 27 bones total:
8 carpals (proximal: scaphoid, lunate, triquetrum, pisiform;
           distal: trapezium, trapezoid, capitate, hamate)
5 metacarpals, 14 phalanges (3 per finger, 2 in thumb)

Functional arches:
1. Proximal transverse arch: rigid, at distal carpal row
2. Distal transverse arch: flexible, at metacarpal heads
3. Longitudinal arch: wrist to fingertip through each ray
Loss of arches in intrinsic paralysis produces flat hand unable to cup objects.

Intrinsic muscles:
Thenar (median): abductor pollicis brevis, flexor pollicis brevis,
opponens pollicis. Hypothenar (ulnar): abductor digiti minimi,
flexor digiti minimi, opponens digiti minimi.
Lumbricals (1,2 median; 3,4 ulnar): MCP flexion with IP extension.
Interossei (dorsal: abduct — DAB; palmar: adduct — PAD).

Flexor tendon pulleys:
A1-A5 annular, C1-C3 cruciate. A2 and A4 most critical — prevent bowstringing.
Pulley injuries: buddy taping, ring splinting, graduated activity.

Grasp patterns:
Power: cylindrical, spherical, hook grasp (ulnar digits)
Precision: tip pinch, lateral/key pinch, 3-jaw chuck (radial/thenar)

Assessment: Jamar dynamometer for grip, pinch gauge for tip/key/palmar pinch.
""",
    "spinal_cord": """
Spinal Cord Anatomy and Clinical Application in OT

The spinal cord extends from foramen magnum to conus medullaris at L1-L2.
Primary conduit for sensory and motor information, contains reflex circuits.

Internal organisation:
Gray matter H-shape: dorsal horn (sensory), ventral horn (motor LMN),
lateral horn (autonomic T1-L2, S2-S4)
White matter: dorsal funiculus, lateral funiculus, anterior funiculus

Ascending tracts:
Dorsal columns: fine touch, two-point discrimination, vibration, proprioception.
Travels ipsilateral → decussates at medulla. Loss = sensory ataxia, Romberg+.
Spinothalamic: pain and temperature. Decussates 1-2 segments above entry.
Brown-Sequard: contralateral pain/temp loss + ipsilateral motor loss.

Descending tracts:
Lateral CST (85-90%): voluntary fine motor, decussates at medulla.
UMN lesion: spasticity, hyperreflexia, Babinski positive.

Key dermatomes for OT:
C5: lateral shoulder, C6: thumb and lateral forearm, C7: middle finger,
C8: little finger and medial forearm, T1: medial arm

Key myotomes:
C5: shoulder abduction, C6: elbow flexion and wrist extension,
C7: elbow extension, C8: finger flexion, T1: finger abduction

ASIA Impairment Scale:
A Complete, B Sensory incomplete, C Motor incomplete (<50% grade 3+),
D Motor incomplete (>50% grade 3+), E Normal

OT functional goals by SCI level:
C4: power wheelchair, adaptive equipment
C5: BFO for feeding, tenodesis development
C6: manual wheelchair, most UE ADLs
C7: full UE function, transfers, homemaking
T1+: full UE independence
""",
}


# ── Image metadata ─────────────────────────────────────────────────────────────

IMAGE_METADATA = [
    {
        "filename": "cerebellum_gray.jpg",
        "source": "Gray's Anatomy",
        "license": "Public Domain",
        "structures": ["cerebellum", "vermis", "cerebellar hemispheres", "folia"],
        "region": "brain",
        "difficulty": "intermediate",
        "ot_relevance": "Motor coordination, ataxia, balance disorders",
    },
    {
        "filename": "brachial_plexus_diagram.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 4.0",
        "structures": [
            "C5-T1 roots",
            "upper trunk",
            "middle trunk",
            "lower trunk",
            "lateral cord",
            "medial cord",
            "posterior cord",
            "median nerve",
            "ulnar nerve",
            "radial nerve",
        ],
        "region": "upper_extremity",
        "difficulty": "advanced",
        "ot_relevance": "Nerve injury, splinting, hand therapy",
    },
    {
        "filename": "hand_bones.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 3.0",
        "structures": [
            "scaphoid",
            "lunate",
            "triquetrum",
            "pisiform",
            "trapezium",
            "trapezoid",
            "capitate",
            "hamate",
            "metacarpals",
            "phalanges",
        ],
        "region": "hand",
        "difficulty": "beginner",
        "ot_relevance": "Hand therapy, fractures, splinting",
    },
    {
        "filename": "cranial_nerves_overview.jpg",
        "source": "OpenStax A&P 2e",
        "license": "CC BY 4.0",
        "structures": ["CN I-XII"],
        "region": "brain",
        "difficulty": "intermediate",
        "ot_relevance": "Dysphagia, facial function, swallowing",
    },
    {
        "filename": "spinal_cord_cross_section.jpg",
        "source": "Wikimedia Commons",
        "license": "Public Domain",
        "structures": [
            "dorsal horn",
            "ventral horn",
            "lateral horn",
            "dorsal columns",
            "corticospinal tract",
            "spinothalamic tract",
        ],
        "region": "spinal_cord",
        "difficulty": "intermediate",
        "ot_relevance": "SCI rehabilitation, sensory assessment",
    },
    {
        "filename": "brain_lobes.jpg",
        "source": "Gray's Anatomy",
        "license": "Public Domain",
        "structures": [
            "frontal lobe",
            "parietal lobe",
            "temporal lobe",
            "occipital lobe",
            "cerebellum",
            "brainstem",
        ],
        "region": "brain",
        "difficulty": "beginner",
        "ot_relevance": "Stroke, TBI, cognitive rehabilitation",
    },
    {
        "filename": "hand_intrinsic_muscles.jpg",
        "source": "Gray's Anatomy",
        "license": "Public Domain",
        "structures": [
            "thenar muscles",
            "hypothenar muscles",
            "lumbricals",
            "interossei",
            "abductor pollicis brevis",
            "opponens pollicis",
        ],
        "region": "hand",
        "difficulty": "advanced",
        "ot_relevance": "Fine motor, intrinsic weakness, ADL",
    },
    {
        "filename": "median_nerve_hand.jpg",
        "source": "Gray's Anatomy",
        "license": "Public Domain",
        "structures": ["median nerve", "carpal tunnel", "thenar branch", "flexor retinaculum"],
        "region": "hand",
        "difficulty": "intermediate",
        "ot_relevance": "Carpal tunnel syndrome, splinting",
    },
    {
        "filename": "shoulder_anatomy.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 3.0",
        "structures": [
            "glenohumeral joint",
            "rotator cuff",
            "supraspinatus",
            "infraspinatus",
            "teres minor",
            "subscapularis",
            "deltoid",
        ],
        "region": "upper_extremity",
        "difficulty": "intermediate",
        "ot_relevance": "Shoulder rehab, overhead activities",
    },
    {
        "filename": "neuron_structure.jpg",
        "source": "Blausen Medical",
        "license": "CC BY 3.0",
        "structures": [
            "cell body",
            "dendrites",
            "axon",
            "myelin sheath",
            "nodes of Ranvier",
            "axon terminals",
        ],
        "region": "nervous_system",
        "difficulty": "beginner",
        "ot_relevance": "Neurological foundations for OT",
    },
    {
        "filename": "dermatomes_anterior.jpg",
        "source": "Grant's Atlas",
        "license": "Public Domain",
        "structures": ["C5-T1 dermatomes", "L1-S3 dermatomes"],
        "region": "spinal_cord",
        "difficulty": "intermediate",
        "ot_relevance": "SCI assessment, sensory testing",
    },
    {
        "filename": "basal_ganglia.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 3.0",
        "structures": [
            "caudate nucleus",
            "putamen",
            "globus pallidus",
            "substantia nigra",
            "subthalamic nucleus",
        ],
        "region": "brain",
        "difficulty": "advanced",
        "ot_relevance": "Parkinson's disease, movement disorders",
    },
    {
        "filename": "vertebral_column.jpg",
        "source": "Gray's Anatomy",
        "license": "Public Domain",
        "structures": [
            "cervical vertebrae",
            "thoracic vertebrae",
            "lumbar vertebrae",
            "sacrum",
            "coccyx",
            "intervertebral discs",
        ],
        "region": "spinal_cord",
        "difficulty": "beginner",
        "ot_relevance": "Posture, ergonomics, SCI",
    },
    {
        "filename": "elbow_joint.jpg",
        "source": "Gray's Anatomy",
        "license": "Public Domain",
        "structures": [
            "humerus",
            "radius",
            "ulna",
            "olecranon",
            "medial epicondyle",
            "lateral epicondyle",
        ],
        "region": "upper_extremity",
        "difficulty": "beginner",
        "ot_relevance": "Elbow rehab, epicondylitis, fractures",
    },
    {
        "filename": "wrist_anatomy.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 3.0",
        "structures": ["radius", "ulna", "scaphoid", "lunate", "radiocarpal joint", "TFCC"],
        "region": "hand",
        "difficulty": "intermediate",
        "ot_relevance": "Wrist splinting, TFCC, fractures",
    },
    {
        "filename": "cerebral_cortex_motor.jpg",
        "source": "OpenStax A&P 2e",
        "license": "CC BY 4.0",
        "structures": [
            "primary motor cortex",
            "somatosensory cortex",
            "premotor cortex",
            "Broca area",
            "Wernicke area",
        ],
        "region": "brain",
        "difficulty": "intermediate",
        "ot_relevance": "Motor learning, stroke, cognitive rehab",
    },
    {
        "filename": "brainstem.jpg",
        "source": "OpenStax A&P 2e",
        "license": "CC BY 4.0",
        "structures": [
            "midbrain",
            "pons",
            "medulla oblongata",
            "corticospinal tract",
            "cranial nerve nuclei",
        ],
        "region": "brain",
        "difficulty": "intermediate",
        "ot_relevance": "Stroke, dysphagia, swallowing",
    },
    {
        "filename": "muscle_fiber_structure.jpg",
        "source": "OpenStax A&P 2e",
        "license": "CC BY 4.0",
        "structures": [
            "muscle fiber",
            "sarcomere",
            "myosin",
            "actin",
            "Z disc",
            "A band",
            "I band",
        ],
        "region": "musculoskeletal",
        "difficulty": "intermediate",
        "ot_relevance": "Muscle physiology, strengthening, fatigue",
    },
    {
        "filename": "ulnar_nerve_path.jpg",
        "source": "Gray's Anatomy",
        "license": "Public Domain",
        "structures": ["ulnar nerve", "cubital tunnel", "Guyon canal", "deep branch", "FCU"],
        "region": "upper_extremity",
        "difficulty": "intermediate",
        "ot_relevance": "Ulnar nerve palsy, claw hand, splinting",
    },
    {
        "filename": "radial_nerve_path.jpg",
        "source": "Gray's Anatomy",
        "license": "Public Domain",
        "structures": [
            "radial nerve",
            "spiral groove",
            "posterior interosseous nerve",
            "extensor muscles",
        ],
        "region": "upper_extremity",
        "difficulty": "intermediate",
        "ot_relevance": "Wrist drop, dynamic splinting",
    },
    {
        "filename": "synapse_diagram.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 3.0",
        "structures": [
            "presynaptic terminal",
            "synaptic vesicles",
            "synaptic cleft",
            "postsynaptic membrane",
            "receptors",
        ],
        "region": "nervous_system",
        "difficulty": "intermediate",
        "ot_relevance": "Neuroplasticity, motor learning",
    },
    {
        "filename": "hip_joint.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 3.0",
        "structures": [
            "femoral head",
            "acetabulum",
            "greater trochanter",
            "femoral neck",
            "hip capsule",
        ],
        "region": "lower_extremity",
        "difficulty": "beginner",
        "ot_relevance": "Hip replacement, transfers, falls",
    },
]


# ── PDF ingestion ──────────────────────────────────────────────────────────────


def ingest_pdf_directly(pdf_path: Path) -> list:
    """
    Read PDF directly and return DocumentChunk list.
    No intermediate .txt files — PDF → chunks in one step.

    Splits by page groups of 5 to create meaningful chunks
    while preserving context across pages.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed — run: pip install pypdf")
        return []

    from src.config.settings import get_settings
    from src.core.rag.chunker import Chunker

    settings = get_settings()
    chunker = Chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if size_mb < 1:
        print(f"  ERROR: PDF is {size_mb:.1f}MB — too small, not a real PDF")
        return []

    print(f"  PDF: {pdf_path.name} ({size_mb:.1f}MB)")

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        print(f"  Pages: {total_pages}")
    except Exception as e:
        print(f"  ERROR reading PDF: {e}")
        return []

    # OT-relevant chapter ranges — focus on nervous system and musculoskeletal
    # OpenStax A&P 2e chapter page ranges (approximate)

    all_chunks = []
    pages_per_group = 8  # Extract 8 pages at a time as one chunk source

    print(f"  Extracting text in groups of {pages_per_group} pages...")
    group_num = 0
    for start in range(0, total_pages, pages_per_group):
        end = min(start + pages_per_group, total_pages)
        group_text = ""

        for page_num in range(start, end):
            try:
                text = reader.pages[page_num].extract_text()
                if text:
                    group_text += text + "\n"
            except Exception:
                continue

        group_text = group_text.strip()
        if len(group_text) < 100:
            continue

        # Detect which chapter this group belongs to
        topic_tag = _detect_chapter_topic(group_text)

        # Chunk this group
        chunks = chunker.chunk_text(
            text=group_text,
            source=f"openStax/anatomy_physiology_2e.pdf#pages_{start + 1}-{end}",
            topic_tags=[topic_tag],
            chapter=f"Pages {start + 1}-{end}",
        )
        all_chunks.extend(chunks)
        group_num += 1

        if group_num % 20 == 0:
            print(
                f"  Processed page group {group_num} ({end}/{total_pages} pages, {len(all_chunks)} chunks so far)..."
            )

    print(f"  Extracted {len(all_chunks)} chunks from {total_pages} pages")
    return all_chunks


def _detect_chapter_topic(text: str) -> str:
    """Detect OT-relevant topic from page text."""
    text_lower = text.lower()
    topic_keywords = {
        "cerebellum": ["cerebellum", "cerebellar", "purkinje", "folia", "vermis"],
        "cranial_nerves": ["cranial nerve", "facial nerve", "vagus", "trigeminal", "hypoglossal"],
        "peripheral_nervous_system": [
            "brachial plexus",
            "median nerve",
            "ulnar nerve",
            "radial nerve",
            "axillary nerve",
        ],
        "spinal_cord": ["spinal cord", "dermatome", "myotome", "corticospinal", "spinothalamic"],
        "hand_anatomy": ["carpal", "metacarpal", "phalanges", "thenar", "intrinsic", "lumbrical"],
        "upper_extremity": ["shoulder", "rotator cuff", "elbow", "biceps", "triceps", "brachialis"],
        "musculoskeletal": [
            "muscle fiber",
            "sarcomere",
            "myosin",
            "actin",
            "motor unit",
            "neuromuscular",
        ],
        "basal_ganglia": [
            "basal ganglia",
            "caudate",
            "putamen",
            "globus pallidus",
            "substantia nigra",
        ],
        "brain_anatomy": [
            "cerebrum",
            "cortex",
            "frontal lobe",
            "parietal",
            "temporal lobe",
            "hippocampus",
        ],
        "nervous_system": ["neuron", "synapse", "action potential", "axon", "dendrite", "myelin"],
    }
    for topic, keywords in topic_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return topic
    return "anatomy_general"


# ── Sample corpus ─────────────────────────────────────────────────────────────


def create_sample_corpus(out_dir: Path) -> list[Path]:
    """
    Write the built-in sample anatomy texts to .txt files in out_dir.
    Returns the list of created file paths. Used for tests and offline
    bootstrapping when the full OpenStax PDF is unavailable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for topic, text in SAMPLE_TEXTS.items():
        path = out_dir / f"{topic}.txt"
        path.write_text(text.strip(), encoding="utf-8")
        files.append(path)
    logger.info("Sample corpus: wrote {n} files to {d}", n=len(files), d=str(out_dir))
    return files


def create_sample_chunks() -> list:
    """Create chunks directly from built-in sample text (no files needed)."""
    from src.config.settings import get_settings
    from src.core.rag.chunker import Chunker

    settings = get_settings()
    chunker = Chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    all_chunks = []
    for topic, text in SAMPLE_TEXTS.items():
        chunks = chunker.chunk_text(
            text=text.strip(),
            source=f"openStax/sample/{topic}.txt",
            topic_tags=[topic],
        )
        all_chunks.extend(chunks)
        logger.info("Sample {t}: {n} chunks", t=topic, n=len(chunks))

    return all_chunks


def create_image_metadata(data_dir: Path) -> Path:
    """Write image_metadata.json."""
    out_path = data_dir / "image_metadata.json"
    out_path.write_text(
        json.dumps(IMAGE_METADATA, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("image_metadata.json: {n} entries", n=len(IMAGE_METADATA))
    return out_path


# ── Vector store ──────────────────────────────────────────────────────────────


async def index_chunks(chunks: list, reset: bool = False) -> int:
    """Embed and index chunks. Returns final chunk count."""
    from src.core.rag.vector_store import get_vector_store

    if not chunks:
        logger.warning("No chunks to index")
        return 0

    store = get_vector_store()

    if reset:
        try:
            await store.reset()
            logger.info("Vector store reset")
        except Exception:
            pass

    # Check for existing data
    try:
        existing = await store.count()
        if existing > 0 and not reset:
            logger.info(
                "Vector store has {n} chunks — resetting to avoid duplicates",
                n=existing,
            )
            await store.reset()
    except Exception:
        pass

    await store.add_chunks(chunks)
    count = await store.count()
    return count


# ── Main pipeline ─────────────────────────────────────────────────────────────


async def run(
    pdf_path: Path | None = None,
    use_sample: bool = False,
    embed_only: bool = False,
    reset: bool = False,
) -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  socratOT — Corpus Ingestion Pipeline")
    print(f"{'=' * 60}\n")

    # ── Image metadata (always update) ────────────────────────────────────
    print("[ 1/3 ] Updating image metadata")
    create_image_metadata(data_dir)
    print(f"  ✓ {len(IMAGE_METADATA)} images documented")

    # ── Build chunks ───────────────────────────────────────────────────────
    chunks = []

    if not embed_only:
        # Try PDF first
        if pdf_path and pdf_path.exists():
            print("\n[ 2/3 ] Reading PDF directly")
            chunks = ingest_pdf_directly(pdf_path)

        # Always add sample chunks on top
        if use_sample or not chunks:
            print("\n[ 2/3 ] Creating sample corpus chunks")
            sample_chunks = create_sample_chunks()
            chunks.extend(sample_chunks)
            print(f"  ✓ {len(sample_chunks)} sample chunks added")

        if not chunks:
            print("  ERROR: No chunks created — check PDF path or use --sample")
            return

        print(f"\n  Total chunks to index: {len(chunks)}")

    else:
        # embed_only: load from saved JSONL if it exists
        chunks_file = ROOT / "data" / "processed" / "chunks" / "openStax_chunks.jsonl"
        if chunks_file.exists():
            import jsonlines as jl

            from src.schemas.rag import DocumentChunk

            with jl.open(chunks_file) as reader:
                chunks = [DocumentChunk(**row) for row in reader]
            print(f"\n[ 2/3 ] Loaded {len(chunks)} existing chunks from JSONL")
        else:
            print("  No saved chunks found — run without --embed-only first")
            return

    # Save chunks to JSONL for future --embed-only runs
    if not embed_only and chunks:
        chunks_dir = ROOT / "data" / "processed" / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunks_dir / "openStax_chunks.jsonl"
        import jsonlines as jl

        with jl.open(chunks_file, mode="w") as writer:
            for c in chunks:
                writer.write(c.model_dump(mode="json"))
        print(f"  ✓ Saved {len(chunks)} chunks to JSONL")

    # ── Embed and index ────────────────────────────────────────────────────
    print("\n[ 3/3 ] Embedding and indexing")
    count = await index_chunks(chunks, reset=reset)
    print(f"  ✓ {count} chunks indexed in ChromaDB")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"  Chunks indexed:  {count}")
    print(f"  Image metadata:  {len(IMAGE_METADATA)} entries")
    print("\n  Next steps:")
    print("  pytest tests/ -v")
    print("  streamlit run app/main.py")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="socratOT corpus ingestion")
    parser.add_argument(
        "--sample", action="store_true", help="Use built-in sample corpus (no PDF needed)"
    )
    parser.add_argument("--pdf", type=str, default=None, help="Path to OpenStax PDF file")
    parser.add_argument(
        "--embed-only", action="store_true", help="Re-embed existing saved chunks only"
    )
    parser.add_argument("--reset", action="store_true", help="Reset vector store before indexing")
    args = parser.parse_args()

    # Auto-detect PDF if not specified
    pdf_path = None
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        default_pdf = ROOT / "data" / "raw" / "openStax" / "anatomy_physiology_2e.pdf"
        if default_pdf.exists() and default_pdf.stat().st_size > 1_000_000:
            pdf_path = default_pdf
            print(f"  Found PDF: {pdf_path} ({pdf_path.stat().st_size / (1024 * 1024):.1f}MB)")

    asyncio.run(
        run(
            pdf_path=pdf_path,
            use_sample=args.sample or pdf_path is None,
            embed_only=args.embed_only,
            reset=args.reset,
        )
    )


if __name__ == "__main__":
    main()
