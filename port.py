import  heapq
import streamlit as st
import simpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time
import random
from collections import defaultdict
import threading
import queue

# Category definitions
class Category(Enum):
    HAZARDOUS = "Hazardous"
    ANIMAL = "Animal"
    FUEL = "Fuel"
    FOOD = "Food"
    CONSUMER = "Consumer"
    CONSTRUCTION = "Construction Supplies"
    EMPTY = "Empty"

@dataclass
class Ship:
    id: int
    category: Category
    arrival_time: float
    service_time: float
    priority: int
    color: str
    size: float  # Visual size factor

# Priority mapping (lower number = higher priority)
CATEGORY_PRIORITY = {
    Category.HAZARDOUS: 0,
    Category.ANIMAL: 1,
    Category.FUEL: 1,
    Category.FOOD: 2,
    Category.CONSUMER: 2,
    Category.CONSTRUCTION: 3,
    Category.EMPTY: 4
}

CATEGORY_COLORS = {
    Category.HAZARDOUS: "#FF0000",  # Red
    Category.ANIMAL: "#3700FF",     # Blue
    Category.FUEL: "#ECF009",       # Yellow
    Category.CONSTRUCTION: "#8B4513", # Brown
    Category.CONSUMER: "#4169E1",   # Blue
    Category.FOOD: "#32CD32",       # Green
    Category.EMPTY: "#A9A9A9"       # Gray
}

class Berth:
    def __init__(self, id: int):
        self.id = id
        self.is_busy = False
        self.current_ship: Optional[Ship] = None
        self.start_time: float = 0

class PriorityQueue:
    """Priority queue for ships using heapq-like behavior"""
    def __init__(self):
        self.ships: List[Tuple[int, float, Ship]] = []  # (priority, arrival_time, ship)
    
    def add(self, ship: Ship):
        heapq.heappush(self.ships, (ship.priority, ship.arrival_time, ship))
    
    def pop(self) -> Optional[Ship]:
        if self.ships:
            return heapq.heappop(self.ships)[2]
        return None
    
    def __len__(self):
        return len(self.ships)
    
    def peek(self) -> Optional[Ship]:
        if self.ships:
            return self.ships[0][2]
        return None

class PortSimulator:
    def __init__(self, arrival_rate: float, num_berths: int, category_probs: Dict[Category, float]):
        self.arrival_rate = arrival_rate
        self.num_berths = num_berths
        self.category_probs = category_probs
        self.categories = list(category_probs.keys())
        
        # Simulation components
        self.env = simpy.Environment()
        self.berths: List[Berth] = [Berth(i) for i in range(num_berths)]
        self.waiting_ships = PriorityQueue()
        self.ship_counter = 0
        
        # Statistics
        self.stats = defaultdict(list)
        self.completed_ships = []
        self.ship_history = []
        
        # Simulation control
        self.running = False
        self.paused = False
        
    def get_category_distribution(self) -> Dict[Category, float]:
        probs = np.random.multinomial(100, list(self.category_probs.values()))
        return {cat: prob/100 for cat, prob in zip(self.category_probs.keys(), probs)}
    
    def generate_service_time(self, category: Category) -> float:
        """Generate service time based on category (in hours)"""
        service_times = {
            Category.HAZARDOUS: (8, 16),    # Slow
            Category.ANIMAL: (1, 3),        # Fast
            Category.FUEL: (6, 12),         # Slow
            Category.FOOD: (2, 4),          # Fast
            Category.CONSUMER: (3, 6),      # Medium
            Category.CONSTRUCTION: (4, 8),  # Medium
            Category.EMPTY: (0.5, 1.5)      # Very Fast
        }
        min_time, max_time = service_times[category]
        return np.random.uniform(min_time, max_time)
    
    def create_ship(self) -> Ship:
        category = np.random.choice(self.categories, p=list(self.category_probs.values()))
        service_time = self.generate_service_time(category)
        size = np.random.uniform(0.8, 1.5)  # Visual size variation
        
        ship = Ship(
            id=self.ship_counter,
            category=category,
            arrival_time=self.env.now,
            service_time=service_time,
            priority=CATEGORY_PRIORITY[category],
            color=CATEGORY_COLORS[category],
            size=size
        )
        self.ship_counter += 1
        return ship
    
    def ship_arrival(self):
        """Ship arrival event"""
        while True:
            yield self.env.timeout(np.random.exponential(1.0 / self.arrival_rate))
            if not self.running:
                break
                
            ship = self.create_ship()
            self.waiting_ships.add(ship)
            self.ship_history.append({
                'time': self.env.now,
                'event': 'arrival',
                'ship_id': ship.id,
                'category': ship.category.value,
                'queue_length': len(self.waiting_ships)
            })
    
    def process_ships(self):
        """Main processing loop - check for available berths"""
        while True:
            available_berth = None
            for berth in self.berths:
                if not berth.is_busy:
                    available_berth = berth
                    break
            
            if available_berth and self.waiting_ships:
                ship = self.waiting_ships.pop()
                self.assign_berth(available_berth, ship)
            
            yield self.env.timeout(0.1)  # Check frequently
    
    def assign_berth(self, berth: Berth, ship: Ship):
        """Assign ship to berth"""
        berth.is_busy = True
        berth.current_ship = ship
        berth.start_time = self.env.now
        
        self.ship_history.append({
            'time': self.env.now,
            'event': 'docking_start',
            'ship_id': ship.id,
            'category': ship.category.value,
            'queue_length': len(self.waiting_ships),
            'berth_id': berth.id
        })
        
        # Schedule departure
        self.env.process(self.service_completion(berth, ship))
    
    def service_completion(self, berth: Berth, ship: Ship):
        """Ship service completion event"""
        wait_time = self.env.now - ship.arrival_time
        service_time = ship.service_time
        
        yield self.env.timeout(service_time)
        
        # Update berth
        berth.is_busy = False
        berth.current_ship = None
        
        # Record statistics
        completion_data = {
            'ship_id': ship.id,
            'category': ship.category.value,
            'arrival_time': ship.arrival_time,
            'docking_time': self.env.now - service_time,
            'departure_time': self.env.now,
            'wait_time': wait_time,
            'service_time': service_time,
            'total_time': wait_time + service_time
        }
        self.completed_ships.append(completion_data)
        
        self.ship_history.append({
            'time': self.env.now,
            'event': 'departure',
            'ship_id': ship.id,
            'category': ship.category.value,
            'queue_length': len(self.waiting_ships),
            'berth_id': berth.id
        })
    
    def run_step(self, delta_time: float = 1.0):
        """Run simulation for delta_time hours"""
        if self.running and not self.paused:
            self.env.step()
    
    def start(self):
        """Start the simulation"""
        self.running = True
        self.env.process(self.ship_arrival())
        self.env.process(self.process_ships())
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
    
    def reset(self):
        """Reset simulation state"""
        self.stop()
        self.env = simpy.Environment()
        self.berths = [Berth(i) for i in range(self.num_berths)]
        self.waiting_ships = PriorityQueue()
        self.ship_counter = 0
        self.completed_ships = []
        self.ship_history = []
        self.stats.clear()

# --- CUSTOM BLACK BACKGROUND CSS ---
st.markdown(
    """
    <style>
    /* Pure black background */
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        background-image: none !important;
        background-attachment: fixed !important;
        background-size: cover !important;
        background-position: center !important;
    }

    /* Black overlay for main container */
    [data-testid="stMainViewContainer"] {
        background-color: #000000 !important;
    }

    /* Glassmorphism effect with black theme */
    .main .block-container {
        background-color: rgba(10, 10, 10, 0.95) !important; 
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.95);
    }

    /* Sidebar styling - pure black */
    [data-testid="stSidebar"] {
        background-color: rgba(5, 5, 5, 0.98) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Global text color adjustments for readability on black */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #F8FAFC !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }

    /* Specific fix for metric labels and values */
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #38BDF8 !important;
    }

    /* Slider and widget contrast improvements */
    .stSlider label {
        color: #CBD5E1 !important;
    }

    /* Button styling for black theme */
    .stButton > button {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #F8FAFC !important;
    }

    .stButton > button:hover {
        background-color: rgba(59, 130, 246, 0.3) !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
    }

    /* Metric container styling */
    [data-testid="stMetric"] {
        background-color: rgba(20, 20, 20, 0.8) !important;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.set_page_config(page_title="Port Operations Simulator", layout="wide")
    st.title("🚢 Port Operations Discrete-Event Simulation")
    st.markdown("**Real-time multi-category ship handling with priority queueing**")
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    # Default category probabilities (must sum to 1.0)
    default_probs = {
        Category.HAZARDOUS: 0.05,
        Category.ANIMAL: 0.10,
        Category.FUEL: 0.15,
        Category.FOOD: 0.20,
        Category.CONSUMER: 0.25,
        Category.CONSTRUCTION: 0.20,
        Category.EMPTY: 0.05
    }
    
    arrival_rate = st.sidebar.slider("Arrival Rate (ships/hour)", 0.5, 5.0, 2.0, 0.1)
    num_berths = st.sidebar.slider("Number of Berths", 1, 8, 3)
    
    st.sidebar.subheader("Category Distribution (%)")
    cat_probs = {}
    total_prob = 0
    for cat in Category:
        prob = st.sidebar.slider(
            f"{cat.value[:3]}",
            0.0, 50.0 - total_prob, 
            default_probs[cat] * 100,
            1.0,
            key=f"prob_{cat.value}"
        ) / 100
        cat_probs[cat] = prob
        total_prob += prob
    
    # Normalize probabilities
    sum_prob = sum(cat_probs.values())
    if sum_prob > 0:
        cat_probs = {cat: prob/sum_prob for cat, prob in cat_probs.items()}
    
    # Simulation speed
    sim_speed = st.sidebar.slider("Simulation Speed (x)", 1, 50, 10)
    
    # Initialize or update simulator
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
        st.session_state.running = False
    
    if st.sidebar.button("Reset Simulation", type="primary"):
        st.session_state.simulator = PortSimulator(arrival_rate, num_berths, cat_probs)
        st.session_state.running = False
        st.rerun()
    
    simulator = st.session_state.simulator
    
    if simulator is None:
        simulator = PortSimulator(arrival_rate, num_berths, cat_probs)
        st.session_state.simulator = simulator
    
    # Update parameters if changed
    simulator.arrival_rate = arrival_rate
    simulator.num_berths = num_berths
    simulator.category_probs = cat_probs
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Start", type="primary", disabled=st.session_state.running):
            simulator.start()
            st.session_state.running = True
    with col2:
        if st.button("⏸️ Pause", disabled=not st.session_state.running):
            simulator.paused = not simulator.paused
    with col3:
        if st.button("⏹️ Stop & Reset"):
            simulator.reset()
            st.session_state.running = False
            st.rerun()
    
    # Real-time simulation loop
    sim_time = st.empty()
    queue_chart = st.empty()
    berth_chart = st.empty()
    stats_chart = st.empty()
    
    # Category legend
    st.markdown("### 🏷️ Ship Categories")
    col_legend = st.columns(len(Category))
    for i, cat in enumerate(Category):
        with col_legend[i]:
            st.markdown(f"""
                <div style="background-color:{CATEGORY_COLORS[cat]}; 
                           width:20px;height:20px;border-radius:50%;display:inline-block;"></div>
                {cat.value}
            """, unsafe_allow_html=True)
    
    if st.session_state.running:
        simulator.start()
        
        # Run simulation steps
        for _ in range(sim_speed):
            simulator.run_step(0.1)
        
        current_time = simulator.env.now
        
        # Main dashboard
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.metric("Simulation Time", f"{current_time:.1f} hours")
            st.metric("Queue Length", len(simulator.waiting_ships))
            st.metric("Ships Completed", len(simulator.completed_ships))
            st.metric("Berths Busy", sum(1 for b in simulator.berths if b.is_busy))
        
        with row1_col2:
            if simulator.completed_ships:
                df_stats = pd.DataFrame(simulator.completed_ships)
                col_stats = st.columns(3)
                with col_stats[0]:
                    st.metric("Avg Wait Time", f"{df_stats['wait_time'].mean():.1f}h")
                with col_stats[1]:
                    st.metric("Avg Service Time", f"{df_stats['service_time'].mean():.1f}h")
                with col_stats[2]:
                    st.metric("Throughput", f"{len(simulator.completed_ships)/current_time:.2f} ships/h")
        
        # Berth visualization
        fig_berths = go.Figure()
        for i, berth in enumerate(simulator.berths):
            y_pos = i + 0.5
            if berth.is_busy and berth.current_ship:
                ship = berth.current_ship
                fig_berths.add_shape(
                    type="rect",
                    x0=0, x1=berth.current_ship.service_time * 2,
                    y0=y_pos-0.4, y1=y_pos+0.4,
                    fillcolor=ship.color,
                    line=dict(color="white", width=2)
                )
                fig_berths.add_annotation(
                    x=1, y=y_pos,
                    text=f"Ship {ship.id}<br>{ship.category.value}",
                    showarrow=False,
                    font=dict(size=10)
                )
            else:
                fig_berths.add_shape(
                    type="rect",
                    x0=0, x1=10,
                    y0=y_pos-0.4, y1=y_pos+0.4,
                    fillcolor="#E0E0E0",
                    line=dict(color="black")
                )
                fig_berths.add_annotation(
                    x=1, y=y_pos,
                    text=f"Berth {i+1}<br>Available",
                    showarrow=False,
                    font=dict(size=10, color="green")
                )
        
        fig_berths.update_layout(
            title="Berth Status",
            xaxis_title="Service Progress",
            yaxis_title="Berths",
            height=300,
            showlegend=False
        )
        berth_chart.plotly_chart(fig_berths, use_container_width=True)
        
        # Detailed statistics
        if simulator.completed_ships:
            df = pd.DataFrame(simulator.completed_ships)
            
            # Category performance
            cat_stats = df.groupby('category').agg({
                'wait_time': ['mean', 'count'],
                'service_time': 'mean',
                'total_time': 'mean'
            }).round(2)
            
            st.subheader("📊 Category Performance")
            st.dataframe(cat_stats, use_container_width=True)
            
            # Waiting time distribution
            fig_wait = px.histogram(
                df, x='wait_time', color='category',
                title="Waiting Time Distribution by Category",
                nbins=20
            )
            st.plotly_chart(fig_wait, use_container_width=True)

if __name__ == "__main__":
    main()
