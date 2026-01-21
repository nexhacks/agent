"""
Formal Police Report Generator

Generates professional law enforcement reports from body camera footage analysis.
Supports multiple report types:
- Traffic Stop Reports (SBI-122 style)
- Officer-Involved Incident Reports (critical incidents)
- General Incident Reports (routine calls)

Reports are generated based on AI analysis of video events, transcripts, and detected actions.
"""

import concurrent.futures
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from video_store import (
    open_video_db,
    get_video_metadata,
    get_all_events,
    get_screenshots_for_video,
)

# Load environment
_env_path = os.path.join(os.path.dirname(__file__), ".env.local")
load_dotenv(_env_path)


class IncidentType(Enum):
    """Types of incidents that can be detected from body cam footage."""
    TRAFFIC_STOP = "traffic_stop"
    OFFICER_INVOLVED_SHOOTING = "officer_involved_shooting"
    USE_OF_FORCE = "use_of_force"
    ARREST = "arrest"
    DOMESTIC_CALL = "domestic_call"
    WELFARE_CHECK = "welfare_check"
    GENERAL_CALL = "general_call"
    UNKNOWN = "unknown"


# Keywords for incident type detection
INCIDENT_KEYWORDS = {
    IncidentType.TRAFFIC_STOP: [
        "traffic stop", "pulled over", "driver", "license", "registration",
        "vehicle", "car", "driving", "speeding", "violation", "citation",
        "driver's window", "driver door", "passenger", "seatbelt"
    ],
    IncidentType.OFFICER_INVOLVED_SHOOTING: [
        "shots fired", "gunshot", "shooting", "firearm discharged",
        "weapon fired", "gun drawn", "gun pointed", "shots", "gunfire"
    ],
    IncidentType.USE_OF_FORCE: [
        "taser", "taser fired", "taser drawn", "physical altercation",
        "resisting", "resistance", "force", "takedown", "restraint",
        "hands behind", "prone", "cuffed", "handcuff"
    ],
    IncidentType.ARREST: [
        "under arrest", "arrested", "custody", "handcuffs", "miranda",
        "rights", "booking", "detained"
    ],
    IncidentType.DOMESTIC_CALL: [
        "domestic", "residence", "apartment", "house", "home",
        "dispute", "argument", "family", "spouse", "partner"
    ],
    IncidentType.WELFARE_CHECK: [
        "welfare check", "wellness", "check on", "concerned",
        "unresponsive", "medical", "injury", "injured"
    ],
}

# Critical event keywords for flagging
CRITICAL_KEYWORDS = [
    "GUN DRAWN", "TASER DRAWN", "TASER FIRED", "SHOTS FIRED",
    "WEAPON!", "GUN VISIBLE", "GUN POINTED",
    "CAMERA BLOCKED", "CAMERA OBSCURED",
    "PERSON ON FLOOR", "PERSON DOWN", "PERSON PRONE",
    "AUDIO: SHOTS FIRED", "AUDIO: TASER FIRED", "AUDIO: EXPLOSION",
]


def format_timestamp(offset_sec: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(offset_sec // 3600)
    minutes = int((offset_sec % 3600) // 60)
    seconds = int(offset_sec % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_time_12hr(dt: datetime) -> str:
    """Format time in 12-hour format."""
    return dt.strftime("%I:%M %p")


def format_date(dt: datetime) -> str:
    """Format date as Month Day, Year."""
    return dt.strftime("%B %d, %Y")


@dataclass
class PersonDescription:
    """Description of a person observed in footage."""
    role: str  # officer, subject, witness, etc.
    sex: Optional[str] = None
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    age_estimate: Optional[str] = None
    clothing: Optional[str] = None
    build: Optional[str] = None
    hair: Optional[str] = None
    distinguishing_features: Optional[str] = None


@dataclass
class VehicleDescription:
    """Description of a vehicle involved in incident."""
    type: str  # car, truck, motorcycle, etc.
    make: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    plate: Optional[str] = None
    year: Optional[str] = None
    condition: Optional[str] = None


@dataclass
class IncidentLocation:
    """Location information for the incident."""
    location_type: str  # street, residence, business, etc.
    address: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    description: Optional[str] = None


@dataclass
class TrafficStopData:
    """Data specific to traffic stop reports (SBI-122 style)."""
    # Initial purpose (check one)
    initial_purpose: str = ""  # checkpoint, violation, DWI, investigation, etc.

    # Driver information
    driver_age: Optional[str] = None
    driver_sex: Optional[str] = None
    driver_race: Optional[str] = None
    driver_ethnicity: Optional[str] = None

    # Enforcement action
    enforcement_action: str = ""  # citation, arrest, warning, no action
    arrest_target: Optional[str] = None  # driver, passenger, both

    # Physical resistance
    physical_resistance: bool = False
    use_of_force: bool = False
    officer_injury: bool = False
    driver_injury: bool = False
    passenger_injury: bool = False

    # Search information
    search_conducted: bool = False
    search_type: Optional[str] = None  # consent, warrant, probable cause, etc.
    search_basis: Optional[str] = None  # suspicious behavior, observation, tip, etc.
    vehicle_searched: bool = False
    driver_searched: bool = False
    passenger_searched: bool = False

    # Contraband
    contraband_found: bool = False
    contraband_type: Optional[str] = None  # drugs, alcohol, weapons, money, other
    contraband_details: Optional[str] = None

    # Property seized
    property_seized: bool = False
    property_details: Optional[str] = None


@dataclass
class OfficerInvolvedIncidentData:
    """Data specific to officer-involved shooting/force reports."""
    # Incident classification
    incident_classification: str = ""  # shooting, use of force, etc.

    # Persons involved
    officer_name: Optional[str] = None
    officer_id: Optional[str] = None
    officer_unit: Optional[str] = None

    subject_name: Optional[str] = None
    subject_status: Optional[str] = None  # deceased, injured, uninjured, fled

    # Weapons/force used
    weapons_used: list = field(default_factory=list)
    force_type: list = field(default_factory=list)

    # Injuries
    officer_injuries: Optional[str] = None
    subject_injuries: Optional[str] = None
    bystander_injuries: Optional[str] = None

    # Evidence
    evidence_collected: list = field(default_factory=list)
    body_cam_footage: bool = True
    witness_statements: bool = False

    # Scene information
    scene_secured: bool = False
    crime_scene_response: bool = False
    medical_response: bool = False


class FormalReportGenerator:
    """Generates formal police reports from video analysis data."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.conn = open_video_db()
        self._client = OpenAI()
        self._video_data = None
        self._events = None
        self._screenshots = None
        self._incident_type = None
        self._analysis_cache = None

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def _load_data(self):
        """Load video data from database."""
        if self._video_data is None:
            self._video_data = get_video_metadata(self.conn, self.video_id)
            if not self._video_data:
                raise ValueError(f"Video not found: {self.video_id}")

        if self._events is None:
            self._events = get_all_events(self.conn, self.video_id)

        if self._screenshots is None:
            self._screenshots = get_screenshots_for_video(self.conn, self.video_id)

    def _get_combined_text(self) -> str:
        """Get all event text combined for analysis."""
        self._load_data()
        texts = []
        for event in self._events:
            texts.append(event["text"].lower())
        return " ".join(texts)

    def _generate_synthesized_timeline(self, interval_sec: float = 10.0) -> list:
        """
        Generate synthesized summaries combining frame analysis and transcripts
        in time intervals (default 10 seconds).

        Returns list of {start_sec, end_sec, timestamp_range, actions, dialogue, synthesis}
        """
        self._load_data()

        if not self._events:
            return []

        # Get video duration
        duration = self._video_data.get("duration_sec", 0) or 0
        if duration <= 0:
            # Estimate from last event
            duration = max(e["offset_sec"] for e in self._events) + interval_sec

        # Group events by interval
        intervals = []
        current_start = 0.0

        while current_start < duration:
            current_end = min(current_start + interval_sec, duration)

            # Get events in this interval
            interval_actions = []
            interval_transcripts = []
            interval_scenes = []

            for event in self._events:
                offset = event["offset_sec"]
                if current_start <= offset < current_end:
                    kind = event["kind"]
                    text = event["text"]

                    if kind == "transcript":
                        # Remove provider prefix
                        for prefix in ["[OpenAI] ", "[Deepgram] "]:
                            if text.startswith(prefix):
                                text = text[len(prefix):]
                                break
                        interval_transcripts.append(text)
                    elif kind == "scene":
                        interval_scenes.append(text)
                    elif kind in ["action", "action_short", "action_long"]:
                        # Filter out "no activity" stubs
                        if text.lower().strip() not in ["no new activity", "no activity", "no change"]:
                            interval_actions.append(text)

            # Format timestamp range
            def fmt_time(sec):
                m = int(sec // 60)
                s = int(sec % 60)
                return f"{m}:{s:02d}"

            timestamp_range = f"{fmt_time(current_start)}-{fmt_time(current_end)}"

            intervals.append({
                "start_sec": current_start,
                "end_sec": current_end,
                "timestamp_range": timestamp_range,
                "actions": interval_actions,
                "dialogue": interval_transcripts,
                "scenes": interval_scenes,
            })

            current_start = current_end

        # Now synthesize each interval using AI
        tasks = [
            (idx, interval)
            for idx, interval in enumerate(intervals)
            if interval["actions"] or interval["dialogue"]
        ]
        if not tasks:
            return []

        synthesized_by_idx = {}
        max_workers = min(4, len(tasks))
        if max_workers <= 1:
            for idx, interval in tasks:
                interval["synthesis"] = self._synthesize_interval(interval)
                synthesized_by_idx[idx] = interval
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._synthesize_interval, interval): idx
                    for idx, interval in tasks
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    synthesis = future.result()
                    interval = intervals[idx]
                    interval["synthesis"] = synthesis
                    synthesized_by_idx[idx] = interval

        return [synthesized_by_idx[idx] for idx, _ in tasks]

    def _synthesize_interval(self, interval: dict) -> str:
        """
        Create a synthesized summary for an interval.

        IMPORTANT: This is FIRST-PERSON body camera footage. The officer wearing
        the camera is NOT visible - only their hands/weapon. Distinguish between
        officer actions (inferred from camera movement, hands visible) and subject
        actions (people visible in frame).
        """
        actions = self._dedupe_strings(interval.get("actions", []))
        dialogue = self._dedupe_strings(interval.get("dialogue", []))
        scenes = self._dedupe_strings(interval.get("scenes", []))
        timestamp = interval.get("timestamp_range", "")

        # If we have very little content, just combine what we have
        if not actions and not dialogue and not scenes:
            return ""

        scene_text = " ".join([f"Scene: {s}" for s in scenes])
        if not actions and not scenes and dialogue:
            return f"Audio: \"{' '.join(dialogue)}\""

        # Check if actions already contain dialogue references (already synthesized)
        if not actions and scenes:
            actions = [f"Scene: {s}" for s in scenes]
        actions_text = " ".join(actions)
        dialogue_keywords = ["said", "shout", "yell", "command", "told", "call", "respond", "asks", "states"]
        already_has_dialogue = any(kw in actions_text.lower() for kw in dialogue_keywords)

        if already_has_dialogue or not dialogue:
            # Actions already incorporate audio context, just return them
            return " ".join([t for t in [actions_text, scene_text] if t]).strip()

        # If we have both separate actions and dialogue, use AI to combine
        prompt = f"""Combine these observations from FIRST-PERSON body camera footage into a detailed 3-4 sentence narrative.

REMEMBER: This is POV from officer's body cam. The OFFICER is the camera wearer (not visible except hands/weapon).
Everyone else in frame is a SUBJECT, CIVILIAN, or BACKUP OFFICER.

VISUAL OBSERVATIONS: {actions_text}
SCENE CONTEXT: {scene_text}

AUDIO TRANSCRIPT: "{' '.join(dialogue)}"

Write a flowing narrative that:
1. Identifies who is speaking (officer giving commands vs subject responding)
2. Describes officer actions (movement, positioning, weapon status)
3. Describes subject actions and compliance level
4. Connects dialogue to physical responses

Use formal report language. Be specific and detailed."""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are writing formal police report narratives from first-person body camera footage. "
                            "The officer wearing the camera is NOT visible except for their hands and weapon. "
                            "Distinguish clearly between officer actions and subject actions. Use third person "
                            "('This officer', 'Subject 1'). Be detailed and precise."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[FormalReportGenerator] Synthesis error: {e}")
            # Fallback: just append dialogue
            return f"{actions_text} Audio: \"{' '.join(dialogue)}\""

    def _generate_full_synthesized_narrative_from_intervals(self, intervals: list) -> str:
        """Generate a detailed narrative from pre-synthesized intervals."""
        if not intervals:
            return "No events captured during this recording."

        narrative_parts = []
        for interval in intervals:
            synthesis = interval.get("synthesis", "")
            if synthesis:
                timestamp = interval.get("timestamp_range", "")
                narrative_parts.append(f"[{timestamp}] {synthesis}")

        return "\n\n".join(narrative_parts)

    def _generate_full_synthesized_narrative(self) -> str:
        """Generate a complete detailed narrative from all synthesized intervals."""
        intervals = self._generate_synthesized_timeline(interval_sec=10.0)
        return self._generate_full_synthesized_narrative_from_intervals(intervals)

    def _sanitize_text(self, text: str) -> str:
        """Strip non-ASCII characters and normalize whitespace."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        cleaned = "".join(ch for ch in text if ord(ch) < 128)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _dedupe_strings(self, items: list) -> list:
        """Return items de-duplicated (case-insensitive), preserving order."""
        seen = set()
        deduped = []
        for item in items or []:
            if not isinstance(item, str):
                item = str(item)
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _format_critical_events(self, critical_events: list) -> str:
        """Format critical events as a readable block."""
        if not critical_events:
            return "None noted."
        lines = []
        for event in critical_events:
            timestamp = self._sanitize_text(event.get("timestamp", ""))
            event_type = self._sanitize_text(event.get("type", ""))
            text = self._sanitize_text(event.get("text", ""))
            lines.append(f"[{timestamp}] {event_type}: {text}".strip())
        return "\n".join(lines)

    def _enrich_narrative(self, analysis: dict, critical_events: list) -> str:
        """Create a concise overview narrative separate from the timeline."""
        base = self._sanitize_text(analysis.get("narrative", ""))
        summary = self._sanitize_text(analysis.get("incident_summary", ""))
        officer_actions = self._sanitize_text(analysis.get("officer_actions", ""))
        subject_actions = self._sanitize_text(analysis.get("subject_actions", ""))
        disposition = self._sanitize_text(analysis.get("disposition", ""))

        parts = []
        if summary:
            parts.append(f"SUMMARY:\n{summary}")
        if base:
            parts.append(f"OVERVIEW:\n{base}")

        word_count = len(re.findall(r"\\b\\w+\\b", base))
        if word_count < 140:
            if officer_actions:
                parts.append(f"OFFICER ACTIONS:\n{officer_actions}")
            if subject_actions:
                parts.append(f"SUBJECT ACTIONS:\n{subject_actions}")
            if disposition:
                parts.append(f"DISPOSITION:\n{disposition}")

        return "\n\n".join(parts).strip()

    def _format_list_items(self, items: list) -> list:
        """Normalize mixed list items to user-friendly strings."""
        formatted = []
        for item in items or []:
            if item is None:
                continue
            if isinstance(item, str):
                text = item.strip()
                if text:
                    formatted.append(text)
                continue
            if isinstance(item, dict):
                item_type = item.get("type") or item.get("category")
                description = item.get("description") or item.get("detail") or item.get("details")
                name = item.get("name")
                if item_type and description:
                    formatted.append(f"{item_type}: {description}")
                elif description:
                    formatted.append(str(description))
                elif name:
                    formatted.append(str(name))
                elif item_type:
                    formatted.append(str(item_type))
                else:
                    formatted.append(json.dumps(item, ensure_ascii=True))
                continue
            formatted.append(str(item))
        return formatted

    def _collect_report_components(self, include_scenes: bool) -> dict:
        """Compute report components concurrently to reduce latency."""
        self._load_data()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                "analysis": executor.submit(self._analyze_incident_with_ai),
                "critical_events": executor.submit(self._extract_critical_events),
                "combined_text": executor.submit(self._get_combined_text),
                "synthesized_timeline": executor.submit(self._generate_synthesized_timeline, 10.0),
            }
            if include_scenes:
                futures["scenes"] = executor.submit(self._get_scene_descriptions)

            results = {key: future.result() for key, future in futures.items()}

        results["synthesized_narrative"] = self._generate_full_synthesized_narrative_from_intervals(
            results["synthesized_timeline"]
        )
        return results

    def detect_incident_type(self) -> IncidentType:
        """Detect the type of incident from video events."""
        if self._incident_type is not None:
            return self._incident_type

        combined_text = self._get_combined_text()

        # Score each incident type
        scores = {}
        for incident_type, keywords in INCIDENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined_text)
            scores[incident_type] = score

        # Check for critical incidents first (they take priority)
        if scores.get(IncidentType.OFFICER_INVOLVED_SHOOTING, 0) >= 1:
            self._incident_type = IncidentType.OFFICER_INVOLVED_SHOOTING
        elif scores.get(IncidentType.USE_OF_FORCE, 0) >= 2:
            self._incident_type = IncidentType.USE_OF_FORCE
        elif scores.get(IncidentType.TRAFFIC_STOP, 0) >= 3:
            self._incident_type = IncidentType.TRAFFIC_STOP
        elif scores.get(IncidentType.ARREST, 0) >= 2:
            self._incident_type = IncidentType.ARREST
        elif scores.get(IncidentType.DOMESTIC_CALL, 0) >= 2:
            self._incident_type = IncidentType.DOMESTIC_CALL
        elif scores.get(IncidentType.WELFARE_CHECK, 0) >= 2:
            self._incident_type = IncidentType.WELFARE_CHECK
        elif max(scores.values(), default=0) > 0:
            # Return the highest scoring type
            self._incident_type = max(scores.keys(), key=lambda k: scores[k])
        else:
            self._incident_type = IncidentType.GENERAL_CALL

        return self._incident_type

    def _extract_critical_events(self) -> list:
        """Extract critical events from the timeline."""
        self._load_data()
        critical = []
        for event in self._events:
            text_upper = event["text"].upper()
            for keyword in CRITICAL_KEYWORDS:
                if keyword in text_upper:
                    critical.append({
                        "offset_sec": event["offset_sec"],
                        "timestamp": format_timestamp(event["offset_sec"]),
                        "type": keyword,
                        "text": event["text"]
                    })
                    break
        return critical

    def _get_scene_descriptions(self) -> list:
        """Get scene description events."""
        self._load_data()
        return [
            {
                "offset_sec": e["offset_sec"],
                "timestamp": format_timestamp(e["offset_sec"]),
                "text": e["text"]
            }
            for e in self._events if e["kind"] == "scene"
        ]

    def _get_transcripts(self) -> str:
        """Get combined transcript text."""
        self._load_data()
        transcripts = []
        for e in self._events:
            if e["kind"] == "transcript":
                # Remove provider prefix
                text = e["text"]
                for prefix in ["[OpenAI] ", "[Deepgram] "]:
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                        break
                transcripts.append(f"[{format_timestamp(e['offset_sec'])}] {text}")
        return "\n".join(transcripts)

    def _analyze_incident_with_ai(self) -> dict:
        """Use AI to analyze the incident and extract structured data."""
        if self._analysis_cache is not None:
            return self._analysis_cache

        self._load_data()
        incident_type = self.detect_incident_type()

        # Build context
        events_text = "\n".join([
            f"[{format_timestamp(e['offset_sec'])}] ({e['kind']}) {e['text']}"
            for e in self._events[:100]  # Limit events
        ])

        scenes = self._get_scene_descriptions()
        scenes_text = "\n".join([f"[{s['timestamp']}] {s['text']}" for s in scenes[:10]])

        critical = self._extract_critical_events()
        critical_text = "\n".join([f"[{c['timestamp']}] {c['type']}: {c['text']}" for c in critical])

        transcript = self._get_transcripts()[:3000]  # Limit length

        prompt = f"""Analyze this FIRST-PERSON body camera footage worn by a police officer and extract detailed information for a formal police report.

IMPORTANT: This is POV footage from the officer's body camera. The camera-wearing officer is NOT visible except for their hands/arms/weapon. Camera movement indicates officer movement.

INCIDENT TYPE DETECTED: {incident_type.value}

SCENE DESCRIPTIONS:
{scenes_text or "None"}

CRITICAL EVENTS:
{critical_text or "None"}

EVENT TIMELINE:
{events_text}

AUDIO TRANSCRIPT:
{transcript or "None"}

Based on this analysis, extract the following information in JSON format:

{{
    "incident_summary": "Comprehensive 4-5 sentence summary covering: what prompted the encounter, key actions by all parties, outcome, and any force used",
    "incident_date_time": "Estimated date/time if mentioned",
    "location": {{
        "type": "street/residence/business/vehicle/other",
        "description": "Detailed description including: street names, address if known, lighting conditions, weather if visible, any notable features",
        "environmental_factors": "Any factors affecting the incident: darkness, confined space, traffic, bystanders"
    }},
    "officer_on_camera": {{
        "identified_by": "How the officer is identifiable (badge number called out, name used, etc.)",
        "equipment_visible": "Equipment observed: weapon type, taser, flashlight, radio",
        "initial_position": "Where officer was positioned at start",
        "movement_summary": "How officer moved throughout incident (approached on foot, exited vehicle, took cover, etc.)"
    }},
    "persons": [
        {{
            "role": "subject/witness/victim/backup_officer",
            "designation": "Subject 1, Subject 2, etc. for tracking",
            "sex": "male/female/unknown",
            "race": "if observable",
            "age_estimate": "approximate age range",
            "height_build": "estimated height and build",
            "clothing": "DETAILED clothing description: colors, types, distinguishing features",
            "hair": "color, length, style",
            "distinguishing_features": "tattoos, glasses, facial hair, injuries visible",
            "initial_position": "Where first observed",
            "demeanor": "cooperative/agitated/compliant/resistant/aggressive",
            "compliance_level": "fully compliant/delayed compliance/verbal resistance/physical resistance/fled",
            "actions_detailed": "Chronological list of their significant actions with approximate timestamps"
        }}
    ],
    "vehicles": [
        {{
            "type": "sedan/suv/truck/motorcycle/van",
            "make": "if identifiable",
            "model": "if identifiable",
            "color": "primary color and any two-tone",
            "license_plate": "if visible",
            "damage_condition": "any visible damage, modifications, or notable condition",
            "position": "where parked/stopped, which direction facing"
        }}
    ],
    "narrative": "WRITE A DETAILED 400-600 WORD NARRATIVE in formal police report style. Include: (1) How the officer arrived on scene and initial observations; (2) Initial contact with subjects - exact positions, what was said (quote key dialogue); (3) Each significant action by the officer with reason/justification; (4) Each response/action by subject(s) with compliance level noted; (5) Any escalation or de-escalation with specific triggers; (6) If force used: exact type, target, effectiveness, subject response; (7) Resolution and final positions/status of all parties. Use third person ('This officer', 'Subject 1'). Be chronologically precise.",
    "officer_actions_detailed": [
        {{
            "timestamp": "approximate time in video",
            "action": "specific action taken",
            "verbal_command": "exact words if command given",
            "justification": "reason for action if applicable"
        }}
    ],
    "subject_actions_detailed": [
        {{
            "timestamp": "approximate time in video",
            "subject": "which subject (Subject 1, etc.)",
            "action": "specific action taken",
            "in_response_to": "what prompted this action if applicable"
        }}
    ],
    "force_used": [
        {{
            "type": "verbal commands/physical control/takedown/taser/OC spray/baton/firearm",
            "timestamp": "when used",
            "target": "which subject",
            "reason": "immediate threat or resistance that prompted use",
            "effectiveness": "subject response to force",
            "injuries_resulted": "any injuries from this force application"
        }}
    ],
    "verbal_exchanges": [
        {{
            "timestamp": "time",
            "speaker": "officer/Subject 1/etc.",
            "statement": "exact or paraphrased words",
            "context": "what prompted this statement"
        }}
    ],
    "injuries": {{
        "officer": "none or detailed description including how sustained",
        "subject": "none or detailed description including how sustained",
        "bystanders": "none or detailed description"
    }},
    "evidence": [
        {{
            "type": "weapon/contraband/documents/vehicle/other",
            "description": "detailed description",
            "location_found": "where observed/recovered",
            "chain_of_custody": "if collected, how secured"
        }}
    ],
    "disposition": "detailed outcome: arrest made (charges), citation issued (violation), warning given, released at scene, medical transport, etc.",
    "recommendations": "any recommended follow-up: investigation, additional charges, victim services, property recovery, etc."
}}

Respond ONLY with valid JSON. Be extremely detailed and specific based on what is observed. The narrative should read like a formal police report that could be submitted to court."""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert police report analyst reviewing FIRST-PERSON body camera footage. "
                            "The footage is from a camera worn on the officer's body - the officer is NOT visible "
                            "except for their hands, arms, and weapon when extended into frame. Camera movement "
                            "indicates the officer's physical movement. Extract accurate, objective, detailed "
                            "information suitable for formal law enforcement reports and court submission. "
                            "Distinguish clearly between officer actions (camera wearer) and subject actions "
                            "(people visible in frame). Always respond with valid JSON only."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2,
            )

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            self._analysis_cache = json.loads(content)
            return self._analysis_cache

        except json.JSONDecodeError as e:
            print(f"[FormalReportGenerator] JSON parse error: {e}")
            return self._get_default_analysis()
        except Exception as e:
            print(f"[FormalReportGenerator] AI analysis error: {e}")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> dict:
        """Return default analysis structure when AI fails."""
        return {
            "incident_summary": "Unable to generate AI summary. Review footage manually.",
            "incident_date_time": "",
            "location": {"type": "unknown", "description": "", "environmental_factors": ""},
            "officer_on_camera": {
                "identified_by": "Unknown",
                "equipment_visible": "Unknown",
                "initial_position": "Unknown",
                "movement_summary": "Review footage"
            },
            "persons": [],
            "vehicles": [],
            "narrative": "AI narrative generation failed. Please review the event timeline and synthesized narrative below for incident details.",
            "officer_actions": "Review timeline for officer actions.",
            "officer_actions_detailed": [],
            "subject_actions": "Review timeline for subject actions.",
            "subject_actions_detailed": [],
            "force_used": [],
            "verbal_exchanges": [],
            "injuries": {"officer": "unknown", "subject": "unknown", "bystanders": "unknown"},
            "evidence": [],
            "disposition": "Pending review",
            "recommendations": ""
        }

    def generate_traffic_stop_report(self) -> dict:
        """Generate a formal Traffic Stop Report (SBI-122 style)."""
        components = self._collect_report_components(include_scenes=False)
        analysis = components["analysis"]

        now = datetime.now()

        # Extract driver info from persons
        driver = None
        officer = None
        for person in analysis.get("persons", []):
            if person.get("role") == "subject":
                driver = person
            elif person.get("role") == "officer":
                officer = person

        # Detect search/contraband from events and analysis
        combined_text = components["combined_text"]
        search_conducted = any(kw in combined_text for kw in ["search", "searched", "searching"])
        contraband_found = any(kw in combined_text for kw in ["contraband", "drugs", "weapon", "gun", "knife"])

        # Detect force used
        force_used = len(analysis.get("force_used", [])) > 0
        critical_events = components["critical_events"]

        narrative = self._enrich_narrative(analysis, critical_events)

        report = {
            "report_type": "TRAFFIC STOP REPORT",
            "form_number": "SBI-122",
            "report_id": f"TS-{self.video_id[:8].upper()}",
            "generated_at": now.isoformat(),

            "header": {
                "agency_name": "[AGENCY NAME]",
                "date": format_date(now),
                "time": format_time_12hr(now),
                "county": analysis.get("location", {}).get("description", "[COUNTY]"),
                "city": "[CITY]",
                "officer_id": "[OFFICER ID]",
            },

            "part_1": {
                "initial_purpose": self._detect_traffic_stop_purpose(combined_text),

                "driver_information": {
                    "age": driver.get("age_estimate") if driver else None,
                    "sex": driver.get("sex") if driver else None,
                    "race": driver.get("race") if driver else None,
                    "ethnicity": None,  # Would need specific detection
                },

                "enforcement_action": self._detect_enforcement_action(analysis),
                "arrest_info": {
                    "arrest_made": "arrest" in combined_text.lower(),
                    "who_arrested": "driver" if "arrest" in combined_text.lower() else None,
                },

                "physical_resistance": {
                    "resistance_encountered": any(kw in combined_text for kw in ["resist", "resisting", "resistance"]),
                    "force_used": force_used,
                    "officer_injured": "officer" in analysis.get("injuries", {}).get("officer", "").lower() if analysis.get("injuries", {}).get("officer") != "none" else False,
                    "driver_injured": analysis.get("injuries", {}).get("subject", "none") != "none",
                    "passenger_injured": False,  # Would need specific detection
                },

                "search_initiated": search_conducted,
            },

            "part_2": {
                "search_type": self._detect_search_type(combined_text) if search_conducted else None,
                "search_basis": self._detect_search_basis(combined_text) if search_conducted else None,

                "persons_vehicle_searched": {
                    "vehicle_searched": "vehicle" in combined_text and search_conducted,
                    "driver_searched": "driver" in combined_text and search_conducted,
                    "passenger_searched": "passenger" in combined_text and search_conducted,
                    "personal_effects_searched": search_conducted,
                },

                "contraband": {
                    "found": contraband_found,
                    "type": analysis.get("evidence", []),
                    "details": None,
                },

                "property_seized": {
                    "seized": any(kw in combined_text for kw in ["seized", "confiscated", "took"]),
                    "type": None,
                    "details": None,
                },
            },

            "incident_summary": analysis.get("incident_summary"),
            "narrative": narrative,
            "officer_actions": analysis.get("officer_actions"),

            "critical_events": critical_events,

            # Synthesized timeline - combines frame analysis with transcripts
            "synthesized_timeline": components["synthesized_timeline"],
            "synthesized_narrative": components["synthesized_narrative"],

            "screenshots": [
                {
                    "id": ss["id"],
                    "timestamp": format_timestamp(ss["offset_sec"]),
                    "offset_sec": ss["offset_sec"],
                    "trigger_type": ss["trigger_type"],
                    "image_data": ss["image_data"]
                }
                for ss in self._screenshots
            ],

            "video_info": {
                "video_id": self.video_id,
                "filename": self._video_data["filename"],
                "duration": format_timestamp(self._video_data["duration_sec"] or 0),
            }
        }

        return report

    def generate_officer_involved_incident_report(self) -> dict:
        """Generate a formal Officer-Involved Incident Report (shooting/use of force)."""
        components = self._collect_report_components(include_scenes=True)
        analysis = components["analysis"]
        incident_type = self.detect_incident_type()

        now = datetime.now()
        critical_events = components["critical_events"]
        scenes = components["scenes"]

        # Determine incident classification
        combined_text = components["combined_text"]
        if incident_type == IncidentType.OFFICER_INVOLVED_SHOOTING:
            classification = "OFFICER INVOLVED SHOOTING"
            file_class = "09005"
        elif incident_type == IncidentType.USE_OF_FORCE:
            classification = "USE OF FORCE INCIDENT"
            file_class = "09003"
        else:
            classification = "OFFICER INVOLVED INCIDENT"
            file_class = "09000"

        # Extract persons
        officers = []
        subjects = []
        witnesses = []
        for person in analysis.get("persons", []):
            if person.get("role") == "officer":
                officers.append(person)
            elif person.get("role") == "subject":
                subjects.append(person)
            elif person.get("role") == "witness":
                witnesses.append(person)

        narrative = self._enrich_narrative(analysis, critical_events)

        report = {
            "report_type": "ORIGINAL INCIDENT REPORT",
            "classification": classification,
            "report_id": f"OIS-{self.video_id[:8].upper()}",
            "incident_number": f"060-{now.strftime('%Y%m%d')}-{self.video_id[:4].upper()}",
            "generated_at": now.isoformat(),

            "header": {
                "department": "[DEPARTMENT NAME]",
                "original_date": format_date(now),
                "time_received": format_time_12hr(now),
                "file_class": file_class,
                "work_unit": "[WORK UNIT]",
                "county": "[COUNTY]",
                "incident_status": "OPEN",
            },

            "complainant": {
                "name": "[PATROL/DISPATCH]",
                "address": "[ADDRESS]",
                "city": "[CITY]",
                "state": "[STATE]",
                "zip": "[ZIP]",
                "telephone": "[PHONE]",
            },

            "incident_title": f"{classification} - {analysis.get('incident_summary', 'PENDING INVESTIGATION')[:50]}",

            "information": {
                "summary": analysis.get("incident_summary"),
                "narrative": narrative,
                "details": f"""On {format_date(now)} at approximately {format_time_12hr(now)}, the following incident occurred:

{narrative or 'See detailed timeline below.'}

OFFICER ACTIONS:
{analysis.get('officer_actions', 'See timeline for details.')}

SUBJECT ACTIONS:
{analysis.get('subject_actions', 'See timeline for details.')}
"""
            },

            "venue": {
                "location_type": analysis.get("location", {}).get("type", "unknown"),
                "description": analysis.get("location", {}).get("description", ""),
                "date_time": {
                    "date": format_date(now),
                    "time": format_time_12hr(now),
                },
                "weather": "[WEATHER CONDITIONS]",
            },

            "officer_involved": [
                {
                    "name": o.get("name", "[OFFICER NAME]"),
                    "badge_number": "[BADGE #]",
                    "sex": o.get("sex"),
                    "race": o.get("race"),
                    "assignment": "[ASSIGNMENT]",
                    "years_service": "[YEARS]",
                    "injuries": analysis.get("injuries", {}).get("officer", "None reported"),
                }
                for o in (officers if officers else [{"name": "[OFFICER NAME]"}])
            ],

            "subject_involved": [
                {
                    "name": s.get("name", "[SUBJECT NAME]") if s.get("name") else "UNIDENTIFIED SUBJECT",
                    "sex": s.get("sex"),
                    "race": s.get("race"),
                    "age": s.get("age_estimate"),
                    "clothing": s.get("clothing"),
                    "build": s.get("build"),
                    "status": analysis.get("disposition", "Unknown"),
                    "injuries": analysis.get("injuries", {}).get("subject", "None reported"),
                }
                for s in (subjects if subjects else [{}])
            ],

            "force_used": {
                "types": [f.get("type", str(f)) if isinstance(f, dict) else str(f) for f in analysis.get("force_used", [])],
                "details": analysis.get("force_used", []),  # Keep full details for reference
                "weapons_discharged": any("shot" in e["type"].lower() or "firearm" in e["type"].lower()
                                         for e in critical_events),
                "taser_deployed": any("taser" in e["type"].lower() for e in critical_events),
                "physical_force": any(kw in combined_text for kw in ["takedown", "restrain", "physical"]),
            },

            "scene_description": {
                "summary": scenes[0]["text"] if scenes else "No scene description available.",
                "all_descriptions": scenes,
            },

            "observations": {
                "initial_scene": scenes[0]["text"] if scenes else "",
                "evidence_observed": analysis.get("evidence", []),
                "body_cam_status": "ACTIVE - Footage captured",
            },

            "evidence_preservation": {
                "body_camera": True,
                "body_camera_status": "Retrieved and secured",
                "in_car_video": "Unknown",
                "security_footage": "Unknown",
                "physical_evidence": analysis.get("evidence", []),
            },

            "vehicles_involved": [
                {
                    "type": v.get("type"),
                    "make": v.get("make"),
                    "model": v.get("model"),
                    "color": v.get("color"),
                    "plate": "[PLATE #]",
                    "vin": "[VIN]",
                }
                for v in analysis.get("vehicles", [])
            ],

            "timeline_of_events": [
                {
                    "timestamp": format_timestamp(e["offset_sec"]),
                    "offset_sec": e["offset_sec"],
                    "event": e["text"],
                    "kind": e["kind"],
                }
                for e in self._events if e["kind"] != "status"
            ],

            "critical_events": critical_events,

            # Synthesized timeline - combines frame analysis with transcripts
            "synthesized_timeline": components["synthesized_timeline"],
            "synthesized_narrative": components["synthesized_narrative"],

            "injuries_summary": analysis.get("injuries", {}),

            "witnesses": [
                {
                    "name": w.get("name", "[WITNESS NAME]"),
                    "contact": "[CONTACT INFO]",
                    "statement_taken": False,
                }
                for w in witnesses
            ],

            "disposition": analysis.get("disposition", "Pending investigation"),
            "recommendations": analysis.get("recommendations", ""),

            "investigating_officer": {
                "name": "[INVESTIGATING OFFICER]",
                "rank": "[RANK]",
                "badge": "[BADGE #]",
            },

            "reviewed_by": {
                "name": "[SUPERVISOR]",
                "rank": "[RANK]",
                "date": format_date(now),
            },

            "screenshots": [
                {
                    "id": ss["id"],
                    "timestamp": format_timestamp(ss["offset_sec"]),
                    "offset_sec": ss["offset_sec"],
                    "trigger_type": ss["trigger_type"],
                    "image_data": ss["image_data"]
                }
                for ss in self._screenshots
            ],

            "video_info": {
                "video_id": self.video_id,
                "filename": self._video_data["filename"],
                "duration": format_timestamp(self._video_data["duration_sec"] or 0),
            },

            "page_info": {
                "total_pages": "1 of 1",
                "printed": now.strftime("%m/%d/%Y %H:%M"),
            }
        }

        return report

    def generate_general_incident_report(self) -> dict:
        """Generate a general incident report for routine calls."""
        components = self._collect_report_components(include_scenes=True)
        analysis = components["analysis"]
        incident_type = self.detect_incident_type()

        now = datetime.now()
        critical_events = components["critical_events"]
        scenes = components["scenes"]

        narrative = self._enrich_narrative(analysis, critical_events)

        report = {
            "report_type": "GENERAL INCIDENT REPORT",
            "incident_type": incident_type.value.replace("_", " ").title(),
            "report_id": f"IR-{self.video_id[:8].upper()}",
            "case_number": f"{now.strftime('%Y')}-{self.video_id[:6].upper()}",
            "generated_at": now.isoformat(),

            "header": {
                "department": "[DEPARTMENT NAME]",
                "date": format_date(now),
                "time": format_time_12hr(now),
                "shift": "[SHIFT]",
                "beat": "[BEAT/ZONE]",
            },

            "reporting_officer": {
                "name": "[OFFICER NAME]",
                "badge_number": "[BADGE #]",
                "unit": "[UNIT]",
            },

            "incident_information": {
                "type": incident_type.value.replace("_", " ").title(),
                "date_occurred": format_date(now),
                "time_occurred": format_time_12hr(now),
                "date_reported": format_date(now),
                "time_reported": format_time_12hr(now),
            },

            "location": {
                "type": analysis.get("location", {}).get("type", "Unknown"),
                "address": "[ADDRESS]",
                "city": "[CITY]",
                "state": "[STATE]",
                "zip": "[ZIP]",
                "description": analysis.get("location", {}).get("description", ""),
            },

            "summary": analysis.get("incident_summary"),

            "narrative": f"""INCIDENT NARRATIVE:

{narrative or 'See event timeline for details.'}

OFFICER ACTIONS:
{analysis.get('officer_actions', 'See timeline.')}

SUBJECT/INVOLVED PARTY ACTIONS:
{analysis.get('subject_actions', 'See timeline.')}

DISPOSITION:
{analysis.get('disposition', 'Pending')}
""",

            "persons_involved": [
                {
                    "role": p.get("role", "Unknown").title(),
                    "name": "[NAME]",
                    "sex": p.get("sex"),
                    "race": p.get("race"),
                    "age": p.get("age_estimate"),
                    "clothing": p.get("clothing"),
                    "description": p.get("build"),
                }
                for p in analysis.get("persons", [])
            ],

            "vehicles_involved": analysis.get("vehicles", []),

            "property_evidence": {
                "items": analysis.get("evidence", []),
                "photos_taken": len(self._screenshots) > 0,
                "photo_count": len(self._screenshots),
            },

            "injuries": analysis.get("injuries", {}),

            "force_used": {
                "force_applied": len(analysis.get("force_used", [])) > 0,
                "types": [f.get("type", str(f)) if isinstance(f, dict) else str(f) for f in analysis.get("force_used", [])],
                "details": analysis.get("force_used", []),  # Keep full details for reference
            },

            "critical_events": critical_events,

            "scene_descriptions": scenes,

            "event_timeline": [
                {
                    "timestamp": format_timestamp(e["offset_sec"]),
                    "event": e["text"],
                }
                for e in self._events if e["kind"] != "status"
            ],

            # Synthesized timeline - combines frame analysis with transcripts
            "synthesized_timeline": components["synthesized_timeline"],
            "synthesized_narrative": components["synthesized_narrative"],

            "disposition": {
                "status": analysis.get("disposition", "Pending"),
                "arrest_made": "arrest" in components["combined_text"].lower(),
                "citation_issued": "citation" in components["combined_text"].lower(),
                "report_taken": True,
                "follow_up_required": bool(analysis.get("recommendations")),
            },

            "recommendations": analysis.get("recommendations", "None"),

            "approving_supervisor": {
                "name": "[SUPERVISOR NAME]",
                "badge": "[BADGE #]",
                "date": format_date(now),
            },

            "screenshots": [
                {
                    "id": ss["id"],
                    "timestamp": format_timestamp(ss["offset_sec"]),
                    "offset_sec": ss["offset_sec"],
                    "trigger_type": ss["trigger_type"],
                    "image_data": ss["image_data"]
                }
                for ss in self._screenshots
            ],

            "video_info": {
                "video_id": self.video_id,
                "filename": self._video_data["filename"],
                "duration": format_timestamp(self._video_data["duration_sec"] or 0),
            }
        }

        return report

    def _detect_traffic_stop_purpose(self, text: str) -> str:
        """Detect the initial purpose of a traffic stop."""
        purposes = {
            "Speed Limit Violation": ["speed", "speeding", "mph", "over the limit"],
            "Stop Light/Sign Violation": ["red light", "stop sign", "ran the"],
            "Driving While Impaired": ["dwi", "dui", "impaired", "intoxicated", "drunk"],
            "Seat Belt Violation": ["seatbelt", "seat belt", "buckled"],
            "Vehicle Equipment Violation": ["tail light", "headlight", "brake light", "equipment"],
            "Vehicle Regulatory Violation": ["registration", "expired", "tag", "plate"],
            "Safe Movement Violation": ["lane", "signal", "turn", "merge"],
            "Investigation": ["match", "suspect", "bolo", "wanted"],
            "Checkpoint": ["checkpoint", "roadblock"],
            "Other Motor Vehicle Violation": ["traffic", "violation"],
        }

        for purpose, keywords in purposes.items():
            if any(kw in text for kw in keywords):
                return purpose

        return "Other Motor Vehicle Violation"

    def _detect_enforcement_action(self, analysis: dict) -> str:
        """Detect the enforcement action taken."""
        disposition = analysis.get("disposition", "").lower()

        if "arrest" in disposition:
            return "On-View Arrest"
        elif "citation" in disposition:
            return "Citation Issued"
        elif "written warning" in disposition:
            return "Written Warning"
        elif "warning" in disposition or "verbal" in disposition:
            return "Verbal Warning"
        elif "no action" in disposition:
            return "No Action Taken"

        return "Citation Issued"  # Default

    def _detect_search_type(self, text: str) -> str:
        """Detect the type of search conducted."""
        if "consent" in text:
            return "Consent"
        elif "warrant" in text:
            return "Search Warrant"
        elif "probable cause" in text or "smell" in text or "plain view" in text:
            return "Probable Cause"
        elif "arrest" in text:
            return "Search Incident to Arrest"
        elif "frisk" in text or "pat down" in text:
            return "Protective Frisk"
        return "Consent"

    def _detect_search_basis(self, text: str) -> str:
        """Detect the basis for the search."""
        if "suspicious" in text or "erratic" in text:
            return "Erratic/Suspicious Behavior"
        elif "smell" in text or "saw" in text or "observed" in text or "plain view" in text:
            return "Observation of Suspected Contraband"
        elif "movement" in text or "reaching" in text:
            return "Suspicious Movement"
        elif "tip" in text or "informant" in text:
            return "Informant's Tip"
        elif "witness" in text:
            return "Witness Observation"
        return "Other Official Information"

    def generate_appropriate_report(self) -> dict:
        """Generate the appropriate report type based on incident detection."""
        incident_type = self.detect_incident_type()

        if incident_type == IncidentType.TRAFFIC_STOP:
            return self.generate_traffic_stop_report()
        elif incident_type in [IncidentType.OFFICER_INVOLVED_SHOOTING, IncidentType.USE_OF_FORCE]:
            return self.generate_officer_involved_incident_report()
        else:
            return self.generate_general_incident_report()

    def to_json(self) -> str:
        """Generate report as JSON string."""
        report = self.generate_appropriate_report()
        return json.dumps(report, indent=2)

    def to_html(self) -> str:
        """Generate report as printable HTML."""
        # Use pre-loaded report data if available, otherwise generate from scratch
        report = getattr(self, '_report_data', None) or self.generate_appropriate_report()
        report_type = report.get("report_type", "INCIDENT REPORT")

        # Get screenshot by ID helper
        screenshots_by_id = {ss["id"]: ss for ss in report.get("screenshots", [])}

        # Start building HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_type} - {report.get('report_id', 'Report')}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 11pt;
            line-height: 1.4;
            max-width: 8.5in;
            margin: 0 auto;
            padding: 0.5in;
            background: white;
            color: #000;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #000;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .header h1 {{
            font-size: 16pt;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .header h2 {{
            font-size: 12pt;
            font-weight: normal;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 5px;
            border: 1px solid #000;
            margin-bottom: 15px;
        }}
        .meta-cell {{
            border: 1px solid #000;
            padding: 3px 5px;
        }}
        .meta-cell label {{
            font-size: 8pt;
            display: block;
            font-weight: bold;
        }}
        .meta-cell value {{
            font-size: 10pt;
        }}
        .section {{
            margin-bottom: 15px;
        }}
        .section-title {{
            font-size: 11pt;
            font-weight: bold;
            text-transform: uppercase;
            background: #f0f0f0;
            padding: 3px 5px;
            border: 1px solid #000;
            margin-bottom: 5px;
        }}
        .section-content {{
            border: 1px solid #000;
            border-top: none;
            padding: 8px;
        }}
        .field-row {{
            display: flex;
            margin-bottom: 5px;
        }}
        .field {{
            flex: 1;
            margin-right: 10px;
        }}
        .field:last-child {{
            margin-right: 0;
        }}
        .field label {{
            font-size: 8pt;
            font-weight: bold;
        }}
        .field value {{
            display: block;
            border-bottom: 1px solid #000;
            min-height: 18px;
            padding: 2px;
        }}
        .narrative {{
            white-space: pre-wrap;
            font-family: 'Times New Roman', Times, serif;
        }}
        .checkbox-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .checkbox-item {{
            display: flex;
            align-items: center;
            gap: 3px;
        }}
        .checkbox {{
            width: 12px;
            height: 12px;
            border: 1px solid #000;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }}
        .checkbox.checked::after {{
            content: "X";
            font-weight: bold;
            font-size: 10px;
        }}
        .critical-alert {{
            background: #fff0f0;
            border: 2px solid #c00;
            padding: 8px;
            margin: 5px 0;
        }}
        .critical-alert .alert-type {{
            color: #c00;
            font-weight: bold;
        }}
        .timeline-entry {{
            padding: 3px 0;
            border-bottom: 1px dotted #ccc;
        }}
        .timeline-entry:last-child {{
            border-bottom: none;
        }}
        .timestamp {{
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            color: #666;
        }}
        .screenshot-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 10px;
        }}
        .screenshot {{
            border: 1px solid #000;
            padding: 5px;
            text-align: center;
        }}
        .screenshot img {{
            max-width: 100%;
            height: auto;
        }}
        .screenshot-meta {{
            font-size: 8pt;
            color: #666;
        }}
        .signature-line {{
            margin-top: 20px;
            padding-top: 10px;
        }}
        .signature-line .line {{
            border-bottom: 1px solid #000;
            width: 250px;
            display: inline-block;
            margin-right: 50px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 5px 0;
        }}
        th, td {{
            border: 1px solid #000;
            padding: 3px 5px;
            text-align: left;
            font-size: 10pt;
        }}
        th {{
            background: #f0f0f0;
            font-size: 9pt;
        }}
        @media print {{
            body {{ padding: 0.25in; }}
            .section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
"""

        # Generate HTML based on report type
        if "TRAFFIC STOP" in report_type:
            html += self._generate_traffic_stop_html(report)
        elif "OFFICER INVOLVED" in report_type or "ORIGINAL INCIDENT" in report_type:
            html += self._generate_ois_html(report)
        else:
            html += self._generate_general_html(report)

        html += """
</body>
</html>
"""
        return html

    def _generate_synthesized_timeline_html(self, report: dict) -> str:
        """Generate HTML for the synthesized timeline section."""
        timeline = report.get("synthesized_timeline", [])
        narrative = report.get("synthesized_narrative", "")
        critical_events = report.get("critical_events", [])

        if not timeline and not narrative:
            return ""

        html = """
<div class="section">
    <div class="section-title">Synthesized Incident Timeline</div>
    <div class="section-content" style="background: #f8fff8; border-left: 4px solid #22c55e;">
        <p style="font-size: 9pt; color: #666; margin-bottom: 10px;">
            <em>AI-synthesized narrative combining visual analysis with audio transcripts in 10-second intervals:</em>
        </p>
"""
        if critical_events:
            html += """
        <div style="margin-bottom: 12px; padding: 8px; background: #fff6f6; border: 1px solid #e0b4b4; border-radius: 4px;">
            <div style="font-weight: bold; color: #991b1b; font-size: 10pt; margin-bottom: 4px;">Critical Events</div>
            <ul style="margin: 4px 0 0 16px; padding: 0; font-size: 9pt;">
"""
            for event in critical_events:
                timestamp = self._sanitize_text(event.get("timestamp", ""))
                event_type = self._sanitize_text(event.get("type", ""))
                text = self._sanitize_text(event.get("text", ""))
                html += f"<li style=\"margin: 2px 0;\">[{timestamp}] {event_type}: {text}</li>"
            html += """
            </ul>
        </div>
"""
        # Show synthesized timeline entries
        for entry in timeline:
            timestamp = entry.get("timestamp_range", "")
            synthesis = entry.get("synthesis", "")
            actions = self._dedupe_strings(entry.get("actions", []))
            dialogue = self._dedupe_strings(entry.get("dialogue", []))
            scenes = self._dedupe_strings(entry.get("scenes", []))
            start_sec = entry.get("start_sec")
            end_sec = entry.get("end_sec")
            entry_screenshots = []
            if start_sec is not None and end_sec is not None:
                for ss in report.get("screenshots", []):
                    offset_sec = ss.get("offset_sec")
                    if offset_sec is None:
                        continue
                    if start_sec <= offset_sec < end_sec:
                        entry_screenshots.append(ss)
                entry_screenshots = entry_screenshots[:3]

            if synthesis or actions or dialogue or scenes:
                timestamp_text = self._sanitize_text(timestamp)
                html += f"""
        <div style="margin-bottom: 12px; padding: 8px; background: white; border-radius: 4px;">
            <div style="font-weight: bold; color: #166534; font-size: 10pt;">[{timestamp_text}]</div>
"""
                if synthesis:
                    synthesis_text = self._sanitize_text(synthesis)
                    html += f"""
            <p style="margin: 6px 0 8px;">{synthesis_text}</p>
"""
                if actions or scenes:
                    html += """
            <div style="font-size: 9pt; color: #444; margin-top: 4px;">
                <strong>Observations:</strong>
                <ul style="margin: 4px 0 0 16px; padding: 0;">
"""
                    for item in actions:
                        html += f"<li style=\"margin: 2px 0;\">{self._sanitize_text(item)}</li>"
                    for item in scenes:
                        html += f"<li style=\"margin: 2px 0;\">Scene: {self._sanitize_text(item)}</li>"
                    html += """
                </ul>
            </div>
"""
                if dialogue:
                    html += """
            <div style="font-size: 9pt; color: #666; margin-top: 6px; padding-left: 10px; border-left: 2px solid #ddd;">
                <strong>Dialogue:</strong>
"""
                    for line in dialogue:
                        html += f"<div style=\"margin-top: 2px;\">\"{self._sanitize_text(line)}\"</div>"
                    html += """
            </div>
"""
                if entry_screenshots:
                    html += """
            <div style="margin-top: 8px;">
                <strong style="font-size: 9pt; color: #444;">Image captures:</strong>
                <div style="display: flex; gap: 8px; margin-top: 6px; flex-wrap: wrap;">
"""
                    for ss in entry_screenshots:
                        html += f"""
                    <div style="border: 1px solid #ccc; padding: 4px; background: #fafafa; text-align: center;">
                        <img src="{ss.get('image_data', '')}" alt="Capture {self._sanitize_text(ss.get('timestamp', ''))}" style="width: 140px; height: auto; display: block;" />
                        <div style="font-size: 8pt; color: #666; margin-top: 4px;">{self._sanitize_text(ss.get('timestamp', ''))}</div>
                    </div>
"""
                    html += """
                </div>
            </div>
"""
                html += "        </div>\n"

        html += """
    </div>
</div>
"""
        return html

    def _generate_traffic_stop_html(self, report: dict) -> str:
        """Generate HTML for traffic stop report."""
        header = report.get("header", {})
        part1 = report.get("part_1", {})
        part2 = report.get("part_2", {})
        driver = part1.get("driver_information", {})
        resistance = part1.get("physical_resistance", {})

        html = f"""
<div class="header">
    <h1>TRAFFIC STOP REPORT</h1>
    <h2>{header.get('agency_name', '[AGENCY NAME]')}</h2>
    <p>Form SBI-122 | Report ID: {report.get('report_id', '')}</p>
</div>

<div class="meta-grid">
    <div class="meta-cell">
        <label>Date</label>
        <value>{header.get('date', '')}</value>
    </div>
    <div class="meta-cell">
        <label>Time</label>
        <value>{header.get('time', '')}</value>
    </div>
    <div class="meta-cell">
        <label>County</label>
        <value>{header.get('county', '')}</value>
    </div>
    <div class="meta-cell">
        <label>Officer ID</label>
        <value>{header.get('officer_id', '')}</value>
    </div>
</div>

<div class="section" style="background: #f0f9ff; border-left: 4px solid #0ea5e9;">
    <div class="section-title">Officer (Camera Wearer) Information</div>
    <div class="section-content">
        <p style="font-size: 9pt; color: #666; margin-bottom: 8px;"><em>Note: This report is generated from first-person body camera footage. The recording officer is not visible except for their hands/weapon.</em></p>
        <div class="field-row">
            <div class="field"><label>Identified By</label><value>{report.get('officer_on_camera', {}).get('identified_by', 'Body camera wearer')}</value></div>
            <div class="field"><label>Equipment Visible</label><value>{report.get('officer_on_camera', {}).get('equipment_visible', 'Standard duty equipment')}</value></div>
        </div>
        <p style="margin-top: 8px;"><strong>Initial Position:</strong> {report.get('officer_on_camera', {}).get('initial_position', 'At scene')}</p>
        <p><strong>Movement Summary:</strong> {report.get('officer_on_camera', {}).get('movement_summary', 'See narrative')}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Part I - Initial Purpose of Traffic Stop</div>
    <div class="section-content">
        <p><strong>Purpose:</strong> {part1.get('initial_purpose', '')}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Driver Information</div>
    <div class="section-content">
        <div class="field-row">
            <div class="field"><label>Age</label><value>{driver.get('age', '')}</value></div>
            <div class="field"><label>Sex</label><value>{driver.get('sex', '')}</value></div>
            <div class="field"><label>Race</label><value>{driver.get('race', '')}</value></div>
            <div class="field"><label>Ethnicity</label><value>{driver.get('ethnicity', '')}</value></div>
        </div>
    </div>
</div>

<div class="section">
    <div class="section-title">Enforcement Action</div>
    <div class="section-content">
        <p><strong>Action Taken:</strong> {part1.get('enforcement_action', '')}</p>
        <p><strong>Arrest Made:</strong> {'Yes' if part1.get('arrest_info', {}).get('arrest_made') else 'No'}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Physical Resistance / Use of Force</div>
    <div class="section-content">
        <div class="checkbox-group">
            <div class="checkbox-item">
                <div class="checkbox {'checked' if resistance.get('resistance_encountered') else ''}"></div>
                <span>Resistance Encountered</span>
            </div>
            <div class="checkbox-item">
                <div class="checkbox {'checked' if resistance.get('force_used') else ''}"></div>
                <span>Force Used</span>
            </div>
            <div class="checkbox-item">
                <div class="checkbox {'checked' if resistance.get('officer_injured') else ''}"></div>
                <span>Officer Injured</span>
            </div>
            <div class="checkbox-item">
                <div class="checkbox {'checked' if resistance.get('driver_injured') else ''}"></div>
                <span>Driver Injured</span>
            </div>
        </div>
    </div>
</div>

<div class="section">
    <div class="section-title">Search Information</div>
    <div class="section-content">
        <p><strong>Search Conducted:</strong> {'Yes' if part1.get('search_initiated') else 'No'}</p>
        {"<p><strong>Search Type:</strong> " + str(part2.get('search_type', '')) + "</p>" if part1.get('search_initiated') else ""}
        {"<p><strong>Search Basis:</strong> " + str(part2.get('search_basis', '')) + "</p>" if part1.get('search_initiated') else ""}
        <p><strong>Contraband Found:</strong> {'Yes' if part2.get('contraband', {}).get('found') else 'No'}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Incident Summary</div>
    <div class="section-content">
        <p>{report.get('incident_summary', '')}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Narrative</div>
    <div class="section-content">
        <div class="narrative">{report.get('narrative', '')}</div>
    </div>
</div>
"""

        # Add synthesized timeline (combines frames + transcripts)
        html += self._generate_synthesized_timeline_html(report)

        # Add screenshots
        html += self._generate_screenshots_html(report.get("screenshots", []))

        # Add signature line
        html += """
<div class="signature-line">
    <p><span class="line"></span> Officer Signature</p>
    <p style="margin-top: 15px;"><span class="line"></span> Supervisor Review</p>
</div>
"""

        return html

    def _generate_ois_html(self, report: dict) -> str:
        """Generate HTML for officer-involved incident report."""
        header = report.get("header", {})
        info = report.get("information", {})
        venue = report.get("venue", {})
        force = report.get("force_used", {})

        html = f"""
<div class="header">
    <h1>{header.get('department', '[DEPARTMENT]')}</h1>
    <h2>{report.get('classification', 'OFFICER INVOLVED INCIDENT')}</h2>
    <p>Incident No: {report.get('incident_number', '')} | Report ID: {report.get('report_id', '')}</p>
</div>

<div class="meta-grid">
    <div class="meta-cell">
        <label>Original Date</label>
        <value>{header.get('original_date', '')}</value>
    </div>
    <div class="meta-cell">
        <label>Time Received</label>
        <value>{header.get('time_received', '')}</value>
    </div>
    <div class="meta-cell">
        <label>File Class</label>
        <value>{header.get('file_class', '')}</value>
    </div>
    <div class="meta-cell">
        <label>Status</label>
        <value>{header.get('incident_status', 'OPEN')}</value>
    </div>
</div>

<div class="section">
    <div class="section-title">Incident Title</div>
    <div class="section-content">
        <p><strong>{report.get('incident_title', '')}</strong></p>
    </div>
</div>

<div class="section" style="background: #f0f9ff; border-left: 4px solid #0ea5e9;">
    <div class="section-title">Recording Officer (Camera Wearer)</div>
    <div class="section-content">
        <p style="font-size: 9pt; color: #666; margin-bottom: 8px;"><em>Note: This report is generated from first-person body camera footage. The recording officer is not visible except for their hands/weapon.</em></p>
        <div class="field-row">
            <div class="field"><label>Identified By</label><value>{report.get('officer_on_camera', {}).get('identified_by', 'Body camera wearer')}</value></div>
            <div class="field"><label>Equipment Visible</label><value>{report.get('officer_on_camera', {}).get('equipment_visible', 'Standard duty equipment')}</value></div>
        </div>
        <p style="margin-top: 8px;"><strong>Initial Position:</strong> {report.get('officer_on_camera', {}).get('initial_position', 'At scene')}</p>
        <p><strong>Movement Summary:</strong> {report.get('officer_on_camera', {}).get('movement_summary', 'See narrative')}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Officer(s) Involved</div>
    <div class="section-content">
        <table>
            <tr>
                <th>Name</th>
                <th>Badge #</th>
                <th>Sex</th>
                <th>Assignment</th>
                <th>Injuries</th>
            </tr>
"""
        for officer in report.get("officer_involved", []):
            html += f"""
            <tr>
                <td>{officer.get('name', '')}</td>
                <td>{officer.get('badge_number', '')}</td>
                <td>{officer.get('sex', '')}</td>
                <td>{officer.get('assignment', '')}</td>
                <td>{officer.get('injuries', 'None')}</td>
            </tr>
"""
        html += """
        </table>
    </div>
</div>

<div class="section">
    <div class="section-title">Subject(s) Involved</div>
    <div class="section-content">
        <table>
            <tr>
                <th>Name/Description</th>
                <th>Sex</th>
                <th>Race</th>
                <th>Age</th>
                <th>Status</th>
                <th>Injuries</th>
            </tr>
"""
        for subject in report.get("subject_involved", []):
            html += f"""
            <tr>
                <td>{subject.get('name', 'Unknown')}<br><small>{subject.get('clothing', '')}</small></td>
                <td>{subject.get('sex', '')}</td>
                <td>{subject.get('race', '')}</td>
                <td>{subject.get('age', '')}</td>
                <td>{subject.get('status', '')}</td>
                <td>{subject.get('injuries', 'None')}</td>
            </tr>
"""
        html += """
        </table>
    </div>
</div>

<div class="section">
    <div class="section-title">Force Used</div>
    <div class="section-content">
        <div class="checkbox-group">
            <div class="checkbox-item">
                <div class="checkbox {'checked' if force.get('weapons_discharged') else ''}"></div>
                <span>Firearm Discharged</span>
            </div>
            <div class="checkbox-item">
                <div class="checkbox {'checked' if force.get('taser_deployed') else ''}"></div>
                <span>Taser Deployed</span>
            </div>
            <div class="checkbox-item">
                <div class="checkbox {'checked' if force.get('physical_force') else ''}"></div>
                <span>Physical Force</span>
            </div>
        </div>
"""
        if force.get("types"):
            force_types = ", ".join(self._format_list_items(force.get("types", [])))
            html += f"""
        <p style="margin-top: 10px;"><strong>Details:</strong> {force_types}</p>
"""
        html += """
    </div>
</div>

<div class="section">
    <div class="section-title">Venue / Scene Description</div>
    <div class="section-content">
        <p><strong>Location Type:</strong> {}</p>
        <p><strong>Description:</strong> {}</p>
    </div>
</div>
""".format(venue.get('location_type', ''), report.get('scene_description', {}).get('summary', ''))

        html += f"""
<div class="section">
    <div class="section-title">Information / Narrative</div>
    <div class="section-content">
        <p><strong>Summary:</strong> {info.get('summary', '')}</p>
        <div class="narrative" style="margin-top: 10px;">
{info.get('details', '')}
        </div>
    </div>
</div>
"""

        # Add synthesized timeline (combines frames + transcripts)
        html += self._generate_synthesized_timeline_html(report)

        # Timeline (abbreviated)
        timeline = report.get("timeline_of_events", [])[:30]
        if timeline:
            html += """
<div class="section">
    <div class="section-title">Event Timeline (Abbreviated)</div>
    <div class="section-content">
"""
            for entry in timeline:
                html += f"""
        <div class="timeline-entry">
            <span class="timestamp">[{entry.get('timestamp', '')}]</span>
            {entry.get('event', '')}
        </div>
"""
            html += "</div></div>"

        # Screenshots
        html += self._generate_screenshots_html(report.get("screenshots", []))

        # Evidence
        evidence = report.get("evidence_preservation", {})
        physical_evidence = ", ".join(
            self._format_list_items(evidence.get("physical_evidence", []))
        ) or "None noted"
        html += f"""
<div class="section">
    <div class="section-title">Evidence Preservation</div>
    <div class="section-content">
        <p><strong>Body Camera:</strong> {'Retrieved and secured' if evidence.get('body_camera') else 'N/A'}</p>
        <p><strong>Physical Evidence:</strong> {physical_evidence}</p>
    </div>
</div>
"""

        # Signatures
        inv = report.get("investigating_officer", {})
        rev = report.get("reviewed_by", {})
        html += f"""
<div class="signature-line">
    <p><strong>Investigating Officer:</strong> {inv.get('name', '')} | {inv.get('rank', '')} | Badge: {inv.get('badge', '')}</p>
    <p style="margin-top: 10px;"><span class="line"></span> Signature</p>
    <p style="margin-top: 20px;"><strong>Reviewed By:</strong> {rev.get('name', '')} | {rev.get('rank', '')} | Date: {rev.get('date', '')}</p>
    <p style="margin-top: 10px;"><span class="line"></span> Supervisor Signature</p>
</div>

<div style="margin-top: 20px; text-align: center; font-size: 9pt; color: #666;">
    <p>{report.get('page_info', {}).get('total_pages', '')} | Printed: {report.get('page_info', {}).get('printed', '')}</p>
</div>
"""

        return html

    def _generate_general_html(self, report: dict) -> str:
        """Generate HTML for general incident report."""
        header = report.get("header", {})
        location = report.get("location", {})
        disposition = report.get("disposition", {})

        html = f"""
<div class="header">
    <h1>{header.get('department', '[DEPARTMENT NAME]')}</h1>
    <h2>GENERAL INCIDENT REPORT</h2>
    <p>Case #: {report.get('case_number', '')} | Report ID: {report.get('report_id', '')}</p>
</div>

<div class="meta-grid">
    <div class="meta-cell">
        <label>Date</label>
        <value>{header.get('date', '')}</value>
    </div>
    <div class="meta-cell">
        <label>Time</label>
        <value>{header.get('time', '')}</value>
    </div>
    <div class="meta-cell">
        <label>Incident Type</label>
        <value>{report.get('incident_type', '')}</value>
    </div>
    <div class="meta-cell">
        <label>Beat/Zone</label>
        <value>{header.get('beat', '')}</value>
    </div>
</div>

<div class="section">
    <div class="section-title">Reporting Officer</div>
    <div class="section-content">
        <div class="field-row">
            <div class="field"><label>Name</label><value>{report.get('reporting_officer', {}).get('name', '')}</value></div>
            <div class="field"><label>Badge #</label><value>{report.get('reporting_officer', {}).get('badge_number', '')}</value></div>
            <div class="field"><label>Unit</label><value>{report.get('reporting_officer', {}).get('unit', '')}</value></div>
        </div>
    </div>
</div>

<div class="section" style="background: #f0f9ff; border-left: 4px solid #0ea5e9;">
    <div class="section-title">Body Camera Analysis</div>
    <div class="section-content">
        <p style="font-size: 9pt; color: #666; margin-bottom: 8px;"><em>Note: This report is generated from first-person body camera footage. The recording officer is not visible except for their hands/weapon.</em></p>
        <div class="field-row">
            <div class="field"><label>Officer Identified By</label><value>{report.get('officer_on_camera', {}).get('identified_by', 'Body camera wearer')}</value></div>
            <div class="field"><label>Equipment Visible</label><value>{report.get('officer_on_camera', {}).get('equipment_visible', 'Standard duty equipment')}</value></div>
        </div>
        <p style="margin-top: 8px;"><strong>Initial Position:</strong> {report.get('officer_on_camera', {}).get('initial_position', 'At scene')}</p>
        <p><strong>Movement During Incident:</strong> {report.get('officer_on_camera', {}).get('movement_summary', 'See narrative')}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Location</div>
    <div class="section-content">
        <div class="field-row">
            <div class="field"><label>Type</label><value>{location.get('type', '')}</value></div>
            <div class="field"><label>Address</label><value>{location.get('address', '')}</value></div>
        </div>
        <p style="margin-top: 5px;"><strong>Description:</strong> {location.get('description', '')}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Summary</div>
    <div class="section-content">
        <p>{report.get('summary', '')}</p>
    </div>
</div>

<div class="section">
    <div class="section-title">Narrative</div>
    <div class="section-content">
        <div class="narrative">{report.get('narrative', '')}</div>
    </div>
</div>
"""

        # Persons involved
        persons = report.get("persons_involved", [])
        if persons:
            html += """
<div class="section">
    <div class="section-title">Persons Involved</div>
    <div class="section-content">
        <table>
            <tr>
                <th>Role</th>
                <th>Name</th>
                <th>Sex</th>
                <th>Race</th>
                <th>Age</th>
                <th>Description</th>
            </tr>
"""
            for person in persons:
                html += f"""
            <tr>
                <td>{person.get('role', '')}</td>
                <td>{person.get('name', '')}</td>
                <td>{person.get('sex', '')}</td>
                <td>{person.get('race', '')}</td>
                <td>{person.get('age', '')}</td>
                <td>{person.get('clothing', '')} {person.get('description', '')}</td>
            </tr>
"""
            html += """
        </table>
    </div>
</div>
"""

        # Force used
        force = report.get("force_used", {})
        if force.get("force_applied"):
            force_types = ", ".join(self._format_list_items(force.get("types", [])))
            html += f"""
<div class="section">
    <div class="section-title">Force Used</div>
    <div class="section-content">
        <p><strong>Types:</strong> {force_types}</p>
    </div>
</div>
"""

        # Add synthesized timeline (combines frames + transcripts)
        html += self._generate_synthesized_timeline_html(report)

        # Disposition
        html += f"""
<div class="section">
    <div class="section-title">Disposition</div>
    <div class="section-content">
        <div class="checkbox-group">
            <div class="checkbox-item">
                <div class="checkbox {'checked' if disposition.get('arrest_made') else ''}"></div>
                <span>Arrest Made</span>
            </div>
            <div class="checkbox-item">
                <div class="checkbox {'checked' if disposition.get('citation_issued') else ''}"></div>
                <span>Citation Issued</span>
            </div>
            <div class="checkbox-item">
                <div class="checkbox {'checked' if disposition.get('report_taken') else ''}"></div>
                <span>Report Taken</span>
            </div>
            <div class="checkbox-item">
                <div class="checkbox {'checked' if disposition.get('follow_up_required') else ''}"></div>
                <span>Follow-up Required</span>
            </div>
        </div>
        <p style="margin-top: 10px;"><strong>Status:</strong> {disposition.get('status', '')}</p>
    </div>
</div>
"""

        # Screenshots
        html += self._generate_screenshots_html(report.get("screenshots", []))

        # Signatures
        supervisor = report.get("approving_supervisor", {})
        html += f"""
<div class="signature-line">
    <p><span class="line"></span> Officer Signature</p>
    <p style="margin-top: 20px;"><strong>Approved By:</strong> {supervisor.get('name', '')} | Badge: {supervisor.get('badge', '')} | Date: {supervisor.get('date', '')}</p>
    <p style="margin-top: 10px;"><span class="line"></span> Supervisor Signature</p>
</div>
"""

        return html

    def _generate_screenshots_html(self, screenshots: list) -> str:
        """Generate HTML for screenshots section."""
        if not screenshots:
            return ""

        html = """
<div class="section">
    <div class="section-title">Body Camera Screenshots</div>
    <div class="section-content">
        <div class="screenshot-grid">
"""
        for ss in screenshots[:12]:  # Limit to 12 screenshots
            html += f"""
            <div class="screenshot">
                <img src="{ss.get('image_data', '')}" alt="Screenshot at {ss.get('timestamp', '')}">
                <div class="screenshot-meta">{ss.get('timestamp', '')} - {ss.get('trigger_type', '')}</div>
            </div>
"""
        html += """
        </div>
    </div>
</div>
"""
        return html


# Convenience functions
def generate_formal_report(video_id: str, format_type: str = "json") -> str:
    """Generate a formal police report for the given video ID."""
    generator = FormalReportGenerator(video_id)

    if format_type == "json":
        return generator.to_json()
    elif format_type == "html":
        return generator.to_html()
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def detect_incident_type(video_id: str) -> str:
    """Detect the incident type for a video."""
    generator = FormalReportGenerator(video_id)
    return generator.detect_incident_type().value
