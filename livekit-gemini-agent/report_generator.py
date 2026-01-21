"""
Police Report Generator

Compiles video analysis data (events, transcripts, screenshots) into
structured reports in JSON and HTML formats with AI-generated summaries.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from video_store import (
    open_video_db,
    get_video_metadata,
    get_all_events,
    get_screenshots_for_video,
    get_twelvelabs_video,
    insert_report_note,
    list_report_notes,
)
from twelvelabs_client import TwelveLabsClient, extract_generated_text

# Load environment for OpenAI key
_env_path = os.path.join(os.path.dirname(__file__), ".env.local")
load_dotenv(_env_path)

TWELVELABS_AUTOFILL = os.getenv("TWELVELABS_AUTOFILL_MISSING", "1") == "1"

# Critical keywords for alert classification
CRITICAL_KEYWORDS = [
    "GUN DRAWN", "TASER DRAWN", "TASER FIRED", "SHOTS FIRED",
    "WEAPON!", "GUN VISIBLE", "GUN POINTED",
    "CAMERA BLOCKED", "CAMERA OBSCURED",
    "PERSON ON FLOOR", "PERSON DOWN", "PERSON PRONE",
    "AUDIO: SHOTS FIRED", "AUDIO: TASER FIRED", "AUDIO: EXPLOSION",
]


def format_timestamp(offset_sec: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(offset_sec // 60)
    seconds = int(offset_sec % 60)
    return f"{minutes:02d}:{seconds:02d}"


def classify_event_tier(kind: str, text: str) -> str:
    """Determine event tier based on kind and content."""
    text_upper = text.upper()

    # Check for critical keywords
    for keyword in CRITICAL_KEYWORDS:
        if keyword in text_upper:
            return "critical"

    # Map by kind
    if kind == "scene":
        return "scene"
    elif kind == "transcript":
        if text.startswith("[OpenAI]"):
            return "transcript_openai"
        elif text.startswith("[Deepgram]"):
            return "transcript_deepgram"
        return "transcript"
    elif kind == "no_activity":
        return "no_activity"
    elif kind == "status":
        return "status"
    elif "warning" in text_upper or "⚠️" in text:
        return "warning"
    else:
        return "action"


def extract_alert_type(text: str) -> Optional[str]:
    """Extract specific alert type from critical event text."""
    text_upper = text.upper()
    for keyword in CRITICAL_KEYWORDS:
        if keyword in text_upper:
            return keyword
    return None


@dataclass
class TimelineEntry:
    offset_sec: float
    timestamp: str
    kind: str
    tier: str
    text: str


@dataclass
class TranscriptSegment:
    offset_sec: float
    timestamp: str
    text: str
    provider: str


@dataclass
class CriticalAlert:
    offset_sec: float
    timestamp: str
    alert_type: str
    text: str
    screenshot_id: Optional[int]


@dataclass
class ScreenshotEntry:
    id: int
    offset_sec: float
    timestamp: str
    trigger_type: str
    image_data_url: str


class ReportGenerator:
    """Generates police reports from video analysis data."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.conn = open_video_db()
        self._client = OpenAI()
        self._cached_report = None  # Cache compiled report to avoid re-computing

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def _generate_ai_summary(self, timeline: list, critical_alerts: list,
                              transcripts: str, scene_descriptions: list,
                              video_duration: float) -> dict:
        """Generate AI-powered summaries for the police report."""
        # Build context for the AI
        timeline_text = "\n".join([
            f"[{e['timestamp']}] ({e['tier'].upper()}) {e['text']}"
            for e in timeline[:50]  # Limit to avoid token overflow
        ])

        critical_text = "\n".join([
            f"[{a['timestamp']}] {a['alert_type']}: {a['text']}"
            for a in critical_alerts
        ]) if critical_alerts else "No critical alerts detected."

        scene_text = "\n".join([
            f"[{s['timestamp']}] {s['text']}"
            for s in scene_descriptions[:5]  # First few scenes
        ]) if scene_descriptions else "No scene descriptions available."

        # Limit transcript length
        transcript_preview = transcripts[:2000] if transcripts else "No transcript available."

        prompt = f"""You are a police report writing assistant. Based on the body camera footage analysis below, generate a professional police report summary.

VIDEO DURATION: {format_timestamp(video_duration or 0)}

SCENE DESCRIPTIONS:
{scene_text}

CRITICAL ALERTS:
{critical_text}

EVENT TIMELINE (chronological):
{timeline_text}

AUDIO TRANSCRIPT:
{transcript_preview}

Generate a police report with the following sections. Write in professional, objective third-person language suitable for official law enforcement documentation:

1. EXECUTIVE SUMMARY (2-3 sentences overview of the incident)

2. INCIDENT NARRATIVE (Detailed chronological account of events in paragraph form, 3-5 paragraphs)

3. KEY OBSERVATIONS (Bullet points of significant findings)

4. PERSONS INVOLVED (Description of individuals observed, their actions and demeanor)

5. OFFICER ACTIONS (Summary of law enforcement actions taken)

6. EVIDENCE & DOCUMENTATION (Note any evidence visible, weapons, injuries observed)

Format your response as JSON with these exact keys:
{{
    "executive_summary": "...",
    "incident_narrative": "...",
    "key_observations": ["...", "..."],
    "persons_involved": "...",
    "officer_actions": "...",
    "evidence_documentation": "..."
}}"""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional police report writer. Generate accurate, objective reports based on body camera footage analysis. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON from response
            # Handle case where response might have markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            summary = json.loads(content)
            return summary

        except json.JSONDecodeError as e:
            print(f"[ReportGenerator] Failed to parse AI response as JSON: {e}")
            # Return a basic summary if JSON parsing fails
            return {
                "executive_summary": "AI summary generation encountered a formatting error. Please review the timeline and events manually.",
                "incident_narrative": "Unable to generate narrative summary. Review the event timeline for details.",
                "key_observations": ["AI summary unavailable - review events manually"],
                "persons_involved": "See scene descriptions for person details.",
                "officer_actions": "Review timeline for officer actions.",
                "evidence_documentation": "Review critical alerts and screenshots."
            }
        except Exception as e:
            print(f"[ReportGenerator] AI summary generation error: {e}")
            return {
                "executive_summary": f"AI summary generation failed: {str(e)}",
                "incident_narrative": "Unable to generate narrative summary.",
                "key_observations": ["AI summary unavailable"],
                "persons_involved": "See scene descriptions.",
                "officer_actions": "Review timeline.",
                "evidence_documentation": "Review screenshots."
            }

    def _identify_missing_questions(self, ai_summary: dict, timeline: list,
                                    scene_descriptions: list, transcripts: str) -> list[str]:
        """Identify missing info that can be answered by additional video analysis."""
        prompt = f"""You are reviewing a draft police report summary and need to find missing factual details.

SUMMARY JSON:
{json.dumps(ai_summary, indent=2)}

TIMELINE SAMPLE:
{json.dumps(timeline[:20], indent=2)}

SCENE DESCRIPTIONS:
{json.dumps(scene_descriptions[:5], indent=2)}

TRANSCRIPT PREVIEW:
{transcripts[:1200] if transcripts else "No transcript available."}

Return a JSON array of up to 3 short, concrete questions about missing facts that could be answered from the video. If nothing is missing, return an empty array. Only return JSON.
"""
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            questions = json.loads(content)
            if isinstance(questions, list):
                return [q for q in questions if isinstance(q, str) and q.strip()]
        except Exception as e:
            print(f"[ReportGenerator] Missing info question generation failed: {e}")
        return []

    def _fill_missing_info(self, ai_summary: dict, timeline: list,
                            scene_descriptions: list, transcripts: str) -> list[dict]:
        if not TWELVELABS_AUTOFILL:
            return []

        twelvelabs = TwelveLabsClient()
        if not twelvelabs.enabled:
            return []

        tl_meta = get_twelvelabs_video(self.conn, self.video_id)
        if not tl_meta or not tl_meta.get("tl_video_id"):
            return []

        if str(tl_meta.get("status", "")).lower() not in {"ready", "indexed", "completed"}:
            return []

        questions = self._identify_missing_questions(
            ai_summary=ai_summary,
            timeline=timeline,
            scene_descriptions=scene_descriptions,
            transcripts=transcripts,
        )
        if not questions:
            return []

        existing_notes = list_report_notes(self.conn, video_id=self.video_id)
        existing_questions = {n["question"].strip().lower() for n in existing_notes}

        additional_info = []
        for question in questions:
            if question.strip().lower() in existing_questions:
                continue
            result = twelvelabs.generate_answer(tl_meta["tl_video_id"], question)
            if not result.ok:
                print(f"[ReportGenerator] TwelveLabs QA failed: {result.error}")
                continue
            answer = extract_generated_text(result.data) or "No answer returned."
            additional_info.append({
                "question": question,
                "answer": answer,
                "source": "twelvelabs_auto",
            })
            insert_report_note(
                self.conn,
                report_id=f"RPT-{self.video_id[:8].upper()}",
                video_id=self.video_id,
                question=question,
                answer=answer,
                source="twelvelabs_auto",
            )

        return additional_info

    def compile_report_data(self) -> dict:
        """Gather all data from database and structure into report.

        Results are cached - calling this multiple times returns the same data.
        """
        # Return cached version if available
        if self._cached_report is not None:
            return self._cached_report

        # Get video metadata
        video = get_video_metadata(self.conn, self.video_id)
        if not video:
            raise ValueError(f"Video not found: {self.video_id}")

        # Get all events
        events = get_all_events(self.conn, self.video_id)

        # Get screenshots
        screenshots = get_screenshots_for_video(self.conn, self.video_id)

        # Build screenshot lookup by approximate time (for matching to events)
        screenshot_by_time = {}
        for ss in screenshots:
            # Round to nearest second for matching
            key = round(ss["offset_sec"])
            screenshot_by_time[key] = ss["id"]

        # Classify and organize events
        timeline = []
        scene_descriptions = []
        transcripts_openai = []
        transcripts_deepgram = []
        critical_alerts = []

        for event in events:
            offset_sec = event["offset_sec"]
            kind = event["kind"]
            text = event["text"]
            tier = classify_event_tier(kind, text)
            timestamp = format_timestamp(offset_sec)

            # Skip status events from report
            if kind == "status":
                continue

            # Add to timeline
            timeline.append({
                "offset_sec": offset_sec,
                "timestamp": timestamp,
                "kind": kind,
                "tier": tier,
                "text": text,
            })

            # Organize by type
            if kind == "scene":
                screenshot_id = screenshot_by_time.get(round(offset_sec))
                scene_descriptions.append({
                    "offset_sec": offset_sec,
                    "timestamp": timestamp,
                    "text": text,
                    "screenshot_id": screenshot_id,
                })

            elif kind == "transcript":
                if text.startswith("[OpenAI]"):
                    transcripts_openai.append({
                        "offset_sec": offset_sec,
                        "timestamp": timestamp,
                        "text": text.replace("[OpenAI] ", ""),
                        "provider": "openai",
                    })
                elif text.startswith("[Deepgram]"):
                    transcripts_deepgram.append({
                        "offset_sec": offset_sec,
                        "timestamp": timestamp,
                        "text": text.replace("[Deepgram] ", ""),
                        "provider": "deepgram",
                    })

            # Check for critical alerts
            if tier == "critical":
                alert_type = extract_alert_type(text) or "ALERT"
                screenshot_id = screenshot_by_time.get(round(offset_sec))
                critical_alerts.append({
                    "offset_sec": offset_sec,
                    "timestamp": timestamp,
                    "alert_type": alert_type,
                    "text": text,
                    "screenshot_id": screenshot_id,
                })

        # Build combined transcript text
        all_transcripts = sorted(
            transcripts_openai + transcripts_deepgram,
            key=lambda x: x["offset_sec"]
        )
        combined_transcript = " ".join(t["text"] for t in all_transcripts)

        # Format screenshots for output
        screenshots_output = []
        for ss in screenshots:
            screenshots_output.append({
                "id": ss["id"],
                "offset_sec": ss["offset_sec"],
                "timestamp": format_timestamp(ss["offset_sec"]),
                "trigger_type": ss["trigger_type"],
                "image_data_url": ss["image_data"],
            })

        # Generate AI summary
        print(f"[ReportGenerator] Generating AI summary for {self.video_id}...")
        ai_summary = self._generate_ai_summary(
            timeline=timeline,
            critical_alerts=critical_alerts,
            transcripts=combined_transcript,
            scene_descriptions=scene_descriptions,
            video_duration=video["duration_sec"] or 0,
        )
        print(f"[ReportGenerator] AI summary generated successfully")

        additional_info = self._fill_missing_info(
            ai_summary=ai_summary,
            timeline=timeline,
            scene_descriptions=scene_descriptions,
            transcripts=combined_transcript,
        )

        # Build final report structure
        report = {
            "report_id": f"RPT-{self.video_id[:8].upper()}",
            "generated_at": datetime.now().isoformat(),
            "video": {
                "id": video["id"],
                "filename": video["filename"],
                "duration_sec": video["duration_sec"],
                "created_at": video["created_at"],
                "status": video["status"],
            },
            "ai_summary": ai_summary,
            "additional_info": additional_info,
            "scene_descriptions": scene_descriptions,
            "critical_alerts": critical_alerts,
            "timeline": timeline,
            "transcripts": {
                "openai": transcripts_openai,
                "deepgram": transcripts_deepgram,
                "combined": combined_transcript,
            },
            "screenshots": screenshots_output,
            "statistics": {
                "total_events": len(timeline),
                "critical_count": len(critical_alerts),
                "scene_count": len(scene_descriptions),
                "transcript_segments": len(all_transcripts),
                "screenshot_count": len(screenshots),
            },
        }

        # Cache the report for subsequent calls
        self._cached_report = report
        return report

    def to_json(self) -> str:
        """Export report as JSON string."""
        report = self.compile_report_data()
        return json.dumps(report, indent=2)

    def to_html(self) -> str:
        """Generate printable HTML report."""
        # Use pre-loaded report data if available, otherwise compile from DB
        report = getattr(self, '_report_data', None) or self.compile_report_data()

        # Get screenshot by ID helper
        screenshots_by_id = {ss["id"]: ss for ss in report["screenshots"]}

        def get_screenshot_img(screenshot_id: Optional[int]) -> str:
            if screenshot_id and screenshot_id in screenshots_by_id:
                ss = screenshots_by_id[screenshot_id]
                return f'<img src="{ss["image_data_url"]}" alt="Screenshot at {ss["timestamp"]}" style="max-width: 400px; border-radius: 4px; margin: 8px 0;">'
            return ""

        # Build HTML
        html_parts = [
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Police Report - {report_id}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .report {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #1a1a2e;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #16213e;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }}
        .header-meta {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }}
        .header-meta p {{
            margin: 5px 0;
        }}
        .critical-section {{
            background: #fff5f5;
            border: 2px solid #dc3545;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .critical-section h2 {{
            color: #dc3545;
            border-bottom-color: #dc3545;
            margin-top: 0;
        }}
        .alert-item {{
            background: white;
            border-left: 4px solid #dc3545;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .alert-type {{
            font-weight: bold;
            color: #dc3545;
        }}
        .timestamp {{
            color: #666;
            font-family: monospace;
            font-size: 0.9em;
        }}
        .scene-item {{
            background: #f0f4ff;
            border-left: 4px solid #6366f1;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .timeline-item {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .timeline-item:last-child {{
            border-bottom: none;
        }}
        .tier-critical {{ color: #dc3545; font-weight: bold; }}
        .tier-warning {{ color: #f59e0b; }}
        .tier-action {{ color: #3b82f6; }}
        .tier-scene {{ color: #6366f1; }}
        .tier-transcript {{ color: #10b981; }}
        .transcript-section {{
            display: flex;
            gap: 20px;
        }}
        .transcript-column {{
            flex: 1;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
        }}
        .transcript-column h3 {{
            margin-top: 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .transcript-column.openai {{ border-top: 3px solid #10b981; }}
        .transcript-column.deepgram {{ border-top: 3px solid #8b5cf6; }}
        .transcript-entry {{
            padding: 5px 0;
            font-size: 0.95em;
        }}
        .screenshot-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }}
        .screenshot-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        .screenshot-item img {{
            max-width: 100%;
            border-radius: 4px;
        }}
        .screenshot-meta {{
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-stat {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }}
        .summary-stat .number {{
            font-size: 2em;
            font-weight: bold;
            color: #1a1a2e;
        }}
        .summary-stat .label {{
            color: #666;
            font-size: 0.9em;
        }}
        .ai-section {{
            background: #f8fffe;
            border: 1px solid #0d9488;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .ai-section h2 {{
            color: #0f766e;
            margin-top: 0;
            border-bottom-color: #0d9488;
        }}
        .ai-section.executive-summary {{
            background: #f0fdf4;
            border-color: #16a34a;
        }}
        .ai-section.executive-summary h2 {{
            color: #15803d;
        }}
        .ai-section.executive-summary .summary-text {{
            font-size: 1.1em;
            font-weight: 500;
            color: #166534;
        }}
        .ai-section.incident-narrative {{
            background: #fefce8;
            border-color: #ca8a04;
        }}
        .ai-section.incident-narrative h2 {{
            color: #a16207;
        }}
        .narrative-text p {{
            margin: 10px 0;
            text-align: justify;
        }}
        .ai-section.key-observations {{
            background: #eff6ff;
            border-color: #3b82f6;
        }}
        .ai-section.key-observations h2 {{
            color: #1d4ed8;
        }}
        .observations-list {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        .observations-list li {{
            margin: 8px 0;
            line-height: 1.5;
        }}
        .ai-section.persons-involved {{
            background: #fdf4ff;
            border-color: #a855f7;
        }}
        .ai-section.persons-involved h2 {{
            color: #7e22ce;
        }}
        .ai-section.officer-actions {{
            background: #f0f9ff;
            border-color: #0ea5e9;
        }}
        .ai-section.officer-actions h2 {{
            color: #0369a1;
        }}
        .ai-section.evidence-documentation {{
            background: #fef2f2;
            border-color: #f87171;
        }}
        .ai-section.evidence-documentation h2 {{
            color: #b91c1c;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.85em;
        }}
        @media print {{
            body {{ background: white; }}
            .report {{ box-shadow: none; padding: 0; }}
            .screenshot-gallery {{ grid-template-columns: repeat(2, 1fr); }}
            .ai-section {{ page-break-inside: avoid; }}
            .ai-section.incident-narrative {{ page-break-inside: auto; }}
        }}
    </style>
</head>
<body>
    <div class="report">
""".format(report_id=report["report_id"])
        ]

        # Header
        html_parts.append(f"""
        <h1>Body Camera Analysis Report</h1>
        <div class="header-meta">
            <p><strong>Report ID:</strong> {report["report_id"]}</p>
            <p><strong>Video File:</strong> {report["video"]["filename"]}</p>
            <p><strong>Duration:</strong> {format_timestamp(report["video"]["duration_sec"] or 0)}</p>
            <p><strong>Recorded:</strong> {report["video"]["created_at"]}</p>
            <p><strong>Report Generated:</strong> {report["generated_at"]}</p>
        </div>
        """)

        # Summary stats
        stats = report["statistics"]
        html_parts.append(f"""
        <div class="summary-grid">
            <div class="summary-stat">
                <div class="number">{stats["critical_count"]}</div>
                <div class="label">Critical Alerts</div>
            </div>
            <div class="summary-stat">
                <div class="number">{stats["scene_count"]}</div>
                <div class="label">Scene Descriptions</div>
            </div>
            <div class="summary-stat">
                <div class="number">{stats["total_events"]}</div>
                <div class="label">Total Events</div>
            </div>
            <div class="summary-stat">
                <div class="number">{stats["screenshot_count"]}</div>
                <div class="label">Screenshots</div>
            </div>
        </div>
        """)

        # AI-Generated Summary Sections
        ai_summary = report.get("ai_summary", {})

        # Executive Summary
        if ai_summary.get("executive_summary"):
            html_parts.append(f"""
        <div class="ai-section executive-summary">
            <h2>Executive Summary</h2>
            <p class="summary-text">{ai_summary["executive_summary"]}</p>
        </div>
            """)

        # Incident Narrative
        if ai_summary.get("incident_narrative"):
            # Convert newlines to paragraphs
            narrative_paragraphs = ai_summary["incident_narrative"].replace("\\n\\n", "</p><p>").replace("\\n", "</p><p>")
            html_parts.append(f"""
        <div class="ai-section incident-narrative">
            <h2>Incident Narrative</h2>
            <div class="narrative-text">
                <p>{narrative_paragraphs}</p>
            </div>
        </div>
            """)

        # Key Observations
        if ai_summary.get("key_observations"):
            observations = ai_summary["key_observations"]
            if isinstance(observations, list):
                obs_html = "".join(f"<li>{obs}</li>" for obs in observations)
            else:
                obs_html = f"<li>{observations}</li>"
            html_parts.append(f"""
        <div class="ai-section key-observations">
            <h2>Key Observations</h2>
            <ul class="observations-list">
                {obs_html}
            </ul>
        </div>
            """)

        # Persons Involved
        if ai_summary.get("persons_involved"):
            html_parts.append(f"""
        <div class="ai-section persons-involved">
            <h2>Persons Involved</h2>
            <p>{ai_summary["persons_involved"]}</p>
        </div>
            """)

        # Officer Actions
        if ai_summary.get("officer_actions"):
            html_parts.append(f"""
        <div class="ai-section officer-actions">
            <h2>Officer Actions</h2>
            <p>{ai_summary["officer_actions"]}</p>
        </div>
            """)

        # Evidence & Documentation
        if ai_summary.get("evidence_documentation"):
            html_parts.append(f"""
        <div class="ai-section evidence-documentation">
            <h2>Evidence & Documentation</h2>
            <p>{ai_summary["evidence_documentation"]}</p>
        </div>
            """)

        # Critical Alerts Section
        if report["critical_alerts"]:
            html_parts.append("""
        <div class="critical-section">
            <h2>Critical Alerts</h2>
            """)
            for alert in report["critical_alerts"]:
                screenshot_html = get_screenshot_img(alert.get("screenshot_id"))
                html_parts.append(f"""
            <div class="alert-item">
                <p><span class="timestamp">[{alert["timestamp"]}]</span> <span class="alert-type">{alert["alert_type"]}</span></p>
                <p>{alert["text"]}</p>
                {screenshot_html}
            </div>
                """)
            html_parts.append("</div>")

        # Scene Descriptions
        if report["scene_descriptions"]:
            html_parts.append("""
        <h2>Scene Descriptions</h2>
            """)
            for scene in report["scene_descriptions"]:
                screenshot_html = get_screenshot_img(scene.get("screenshot_id"))
                html_parts.append(f"""
        <div class="scene-item">
            <p><span class="timestamp">[{scene["timestamp"]}]</span></p>
            <p>{scene["text"]}</p>
            {screenshot_html}
        </div>
                """)

        # Timeline
        html_parts.append("""
        <h2>Event Timeline</h2>
        """)
        for entry in report["timeline"]:
            tier_class = f"tier-{entry['tier'].split('_')[0]}"
            html_parts.append(f"""
        <div class="timeline-item">
            <span class="timestamp">[{entry["timestamp"]}]</span>
            <span class="{tier_class}">[{entry["kind"].upper()}]</span>
            {entry["text"]}
        </div>
            """)

        # Transcripts
        html_parts.append("""
        <h2>Transcripts</h2>
        <div class="transcript-section">
            <div class="transcript-column openai">
                <h3>OpenAI Whisper</h3>
        """)
        for t in report["transcripts"]["openai"]:
            html_parts.append(f"""
                <div class="transcript-entry">
                    <span class="timestamp">[{t["timestamp"]}]</span> {t["text"]}
                </div>
            """)
        html_parts.append("""
            </div>
            <div class="transcript-column deepgram">
                <h3>Deepgram</h3>
        """)
        for t in report["transcripts"]["deepgram"]:
            html_parts.append(f"""
                <div class="transcript-entry">
                    <span class="timestamp">[{t["timestamp"]}]</span> {t["text"]}
                </div>
            """)
        html_parts.append("""
            </div>
        </div>
        """)

        # Screenshots Gallery
        if report["screenshots"]:
            html_parts.append("""
        <h2>Screenshot Gallery</h2>
        <div class="screenshot-gallery">
            """)
            for ss in report["screenshots"]:
                html_parts.append(f"""
            <div class="screenshot-item">
                <img src="{ss["image_data_url"]}" alt="Screenshot at {ss["timestamp"]}">
                <div class="screenshot-meta">
                    {ss["timestamp"]} - {ss["trigger_type"]}
                </div>
            </div>
                """)
            html_parts.append("</div>")

        # Footer
        html_parts.append(f"""
        <div class="footer">
            <p>Generated by Clearance Body Cam Analysis System</p>
            <p>Report ID: {report["report_id"]} | Generated: {report["generated_at"]}</p>
        </div>
    </div>
</body>
</html>
        """)

        return "".join(html_parts)


# Convenience function for direct use
def generate_report(video_id: str, format_type: str = "json") -> str | bytes:
    """Generate a report for the given video ID in the specified format."""
    generator = ReportGenerator(video_id)

    if format_type == "json":
        return generator.to_json()
    elif format_type == "html":
        return generator.to_html()
    else:
        raise ValueError(f"Unsupported format: {format_type}")
