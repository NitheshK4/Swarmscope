from abc import ABC, abstractmethod
from sandbox.schemas import SimulationRun, SimulationMetadata, Message

class BaseStore(ABC):
    @abstractmethod
    def save_run(self, run: SimulationRun) -> None:
        """Saves a full simulation run, including metadata, messages, and detector scores."""
        pass

    @abstractmethod
    def get_run(self, simulation_id: str) -> SimulationRun:
        """Retrieves a single simulation run by its ID."""
        pass

    @abstractmethod
    def get_all_runs(self) -> list[SimulationMetadata]:
        """Retrieves metadata for all runs stored in the database."""
        pass

    @abstractmethod
    def get_run_messages(self, simulation_id: str) -> list[Message]:
        """Retrieves all message logs associated with a given simulation ID."""
        pass
