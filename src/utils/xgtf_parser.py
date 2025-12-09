"""
Parses ground truth's from MIVIA dataset

Retrieves crossing events and corresponding frame numebers, 
person IDs, crossing direction, and crossing configuration

Insight on configurations avaiable at data/mivia/config.txt
"""

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

@dataclass
class CrossingEvent:
    """Information related to a single crossing event"""
    frame: int
    person_id: int
    dir: str
    configuration: str
    
@dataclass
class VideoMetadata:
    """Video metadata related to ground truth file"""
    num_frames: int
    frame_rate: float
    width: int
    height: int

    @property
    def video_playtime(self) -> float:
        """Calculates video duration in seconds"""
        return self.num_frames / self.frame_rate

@dataclass
class GroundTruth:
    """Contains parsed ground truth data from a single XGTF file"""
    filename: str
    metadata: VideoMetadata
    events: list[CrossingEvent] = field(default_factory=list) # New list per object

    @property
    def total_crossings(self) -> int:
        """Total number of crossing events"""
        return len(self.events)
    
    def get_counts_by_dir(self) -> dict[str, int]:
        """Count crossings per direction"""
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.dir] = counts.get(event.dir, 0) + 1
        return counts
    
def parse_xgtf(filepath: Path) -> GroundTruth:
    """Parse XGTF ground truth file"""
    
    if not filepath.exists(): 
        raise FileNotFoundError(f"XGTF file not found: {filepath}")
    
    # Parse XML with namespace handling
    tree = ET.parse(filepath)
    root = tree.getroot()
    namespaces = {
        'viper': 'http://lamp.cfar.umd.edu/viper#',
        'data': 'http://lamp.cfar.umd.edu/viperdata#'
    }

    # Extract Metadata
    metadata = VideoMetadata()
    file_elem = root.find('.//viper:data/viper:sourcefile/viper:file', namespaces)

    if file_elem is not None:
        numframes = file_elem.find('viper:attribute[@name="NUMFRAMES"]/data:dvalue', namespaces)
        if numframes is not None and 'value' in numframes.attrib:
            metadata.num_frames = int(numframes.attrib['value'])
        
        framerate = file_elem.find('viper:attribute[@name="FRAMERATE"]/data:fvalue', namespaces)
        if framerate is not None and 'value' in framerate.attrib:
            metadata.framerate = float(framerate.attrib['value'])
        
        width = file_elem.find('viper:attribute[@name="H-FRAME-SIZE"]/data:dvalue', namespaces)
        if width is not None and 'value' in width.attrib:
            metadata.width = int(width.attrib['value'])
        
        height = file_elem.find('viper:attribute[@name="V-FRAME-SIZE"]/data:dvalue', namespaces)
        if height is not None and 'value' in height.attrib:
            metadata.height = int(height.attrib['value'])
    
    # Extract crossing events
    events: list[CrossingEvent] = []
    objects = root.findall('.//viper:data/viper:sourcefile/viper:object[@name="PERSON"]', namespaces)
    
    for obj in objects:
        framespan = obj.attrib.get('framespan')
        frame = int(framespan.split(':')[0])
        
        person_id = int(obj.attrib.get('id'))
        
        direction_elem = obj.find('viper:attribute[@name="Crossing"]/data:lvalue', namespaces)
        direction = direction_elem.attrib.get('value')
        
        config_elem = obj.find('viper:attribute[@name="Crossing Configuration"]/data:lvalue', namespaces)
        configuration = config_elem.attrib.get('value')
        
        events.append(CrossingEvent(
            frame=frame,
            person_id=person_id,
            direction=direction,
            configuration=configuration
        ))
    
    # Sort by frame number
    events.sort(key=lambda e: (e.frame, e.person_id))
    
    return GroundTruth(
        filename=filepath.stem,
        metadata=metadata,
        events=events
    )


def load_all_ground_truths(folder: Path) -> dict[str, GroundTruth]:
    """Load all XGTF ground truth files from a folder."""

    ground_truths: dict[str, GroundTruth] = {}
    
    for xgtf_file in sorted(folder.glob("*.xgtf")):
        try:
            gt = parse_xgtf(xgtf_file)
            ground_truths[gt.filename] = gt
        except Exception as e:
            print(f"Warning: Failed to parse {xgtf_file.name}: {e}")
    
    return ground_truths