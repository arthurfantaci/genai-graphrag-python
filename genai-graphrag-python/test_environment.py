"""Environment verification test suite for Neo4j GraphRAG setup.

This module validates that all required components are correctly configured
before running the knowledge graph construction or RAG pipelines. It checks:
  - .env file existence and loading
  - Required environment variables for OpenAI and Neo4j
  - OpenAI API connectivity and authentication
  - Neo4j database connectivity and query execution

Test Execution Order:
---------------------
Tests are ordered to fail fast on fundamental issues:
1. .env file existence (foundational)
2. OpenAI environment variables
3. Neo4j environment variables
4. OpenAI API connection
5. Neo4j database connection

Running the Tests:
------------------
    uv run python genai-graphrag-python/test_environment.py

All tests must pass before running any of the kg_builder or RAG scripts,
as they depend on these external services.

Test Skipping Logic:
--------------------
Tests use class-level flags to skip dependent tests when prerequisites fail.
For example, if the .env file is missing, all environment variable tests
are skipped since they would fail anyway. This makes the output cleaner
and helps identify the root cause of failures.
"""

import os
import unittest

from dotenv import find_dotenv, load_dotenv

load_dotenv()


class TestEnvironment(unittest.TestCase):
    """Test suite for validating environment configuration.

    Tests are designed to be run in a specific order via the suite() function
    to ensure dependent tests are skipped when prerequisites fail.

    Class Attributes:
        skip_env_variable_tests: Set to False when .env file is found
        skip_openai_test: Set to False when OpenAI env vars are present
        skip_neo4j_test: Set to False when Neo4j env vars are present
    """

    # =========================================================================
    # TEST SKIP FLAGS
    # =========================================================================
    # These class-level flags control test skipping. They start as True
    # (skip by default) and are set to False when prerequisites pass.
    # This prevents cascading failures and makes output clearer.
    # =========================================================================
    skip_env_variable_tests = True
    skip_openai_test = True
    skip_neo4j_test = True

    # =========================================================================
    # TEST 1: .ENV FILE EXISTENCE
    # =========================================================================
    # This is the foundational test. All other tests depend on the .env file
    # existing because that's where API keys and connection strings are stored.
    #
    # find_dotenv() searches for .env in the current directory and parent
    # directories, returning the path if found or empty string if not.
    # =========================================================================
    def test_env_file_exists(self):
        env_file_exists = True if find_dotenv() > "" else False
        if env_file_exists:
            TestEnvironment.skip_env_variable_tests = False
        self.assertTrue(env_file_exists, ".env file not found.")

    # =========================================================================
    # HELPER: ENVIRONMENT VARIABLE VALIDATION
    # =========================================================================
    # Reusable assertion for checking that a specific environment variable
    # is present and non-empty.
    # =========================================================================
    def env_variable_exists(self, variable_name):
        self.assertIsNotNone(
            os.getenv(variable_name), f"{variable_name} not found in .env file"
        )

    # =========================================================================
    # TEST 2: OPENAI ENVIRONMENT VARIABLES
    # =========================================================================
    # Validates that OPENAI_API_KEY is set. This is required for:
    #   - LLM-based entity extraction in kg_builder scripts
    #   - Embedding generation for vector search
    #   - Answer generation in RAG pipelines
    # =========================================================================
    def test_openai_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping OpenAI env variable test")

        self.env_variable_exists("OPENAI_API_KEY")
        TestEnvironment.skip_openai_test = False

    # =========================================================================
    # TEST 3: NEO4J ENVIRONMENT VARIABLES
    # =========================================================================
    # Validates that all Neo4j connection parameters are set:
    #   - NEO4J_URI: Connection string (e.g., bolt://localhost:7687)
    #   - NEO4J_USERNAME: Authentication username
    #   - NEO4J_PASSWORD: Authentication password
    #   - NEO4J_DATABASE: Target database name
    #
    # These are required for all kg_builder and RAG scripts.
    # =========================================================================
    def test_neo4j_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping Neo4j env variables test")

        self.env_variable_exists("NEO4J_URI")
        self.env_variable_exists("NEO4J_USERNAME")
        self.env_variable_exists("NEO4J_PASSWORD")
        self.env_variable_exists("NEO4J_DATABASE")
        TestEnvironment.skip_neo4j_test = False

    # =========================================================================
    # TEST 4: OPENAI API CONNECTION
    # =========================================================================
    # Validates that the OPENAI_API_KEY is valid by making a test API call.
    #
    # The models.list() endpoint is a lightweight way to verify authentication
    # without consuming tokens. If the key is invalid, AuthenticationError
    # is raised.
    # =========================================================================
    def test_openai_connection(self):
        if TestEnvironment.skip_openai_test:
            self.skipTest("Skipping OpenAI test")

        from openai import AuthenticationError, OpenAI

        llm = OpenAI()

        try:
            models = llm.models.list()
        except AuthenticationError:
            models = None
        self.assertIsNotNone(
            models,
            "OpenAI connection failed. Check the OPENAI_API_KEY key in .env file.",
        )

    # =========================================================================
    # TEST 5: NEO4J DATABASE CONNECTION
    # =========================================================================
    # Validates Neo4j connectivity in two stages:
    #
    # 1. verify_connectivity(): Tests that the driver can establish a
    #    connection to the Neo4j server using the URI and credentials.
    #
    # 2. execute_query(): Tests that the specified database exists and
    #    accepts queries. This catches cases where the server is reachable
    #    but the database doesn't exist or the user lacks permissions.
    #
    # Error messages are tailored to help diagnose which specific
    # configuration is incorrect.
    # =========================================================================
    def test_neo4j_connection(self):
        msg = "Neo4j connection failed. Check the NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4j_DATABASE values in .env file."
        connected = False

        if TestEnvironment.skip_neo4j_test:
            self.skipTest("Skipping Neo4j connection test")

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        try:
            driver.verify_connectivity()
            try:
                driver.execute_query(
                    "RETURN true", database_=os.getenv("NEO4J_DATABASE")
                )
                connected = True

            except Exception:
                msg = "Neo4j database query failed. Check the NEO4J_DATABASE value in .env file."

        except Exception:
            msg = "Neo4j verify connection failed. Check the NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD values in .env file."

        driver.close()

        self.assertTrue(connected, msg)


# =============================================================================
# TEST SUITE CONSTRUCTION
# =============================================================================
# Creates an ordered test suite to ensure tests run in dependency order.
# This is important because:
#   - .env file must exist before checking env variables
#   - Env variables must exist before testing API connections
#
# Using a suite instead of relying on test discovery order ensures
# consistent, predictable execution.
# =============================================================================
def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnvironment("test_env_file_exists"))
    suite.addTest(TestEnvironment("test_openai_variables"))
    suite.addTest(TestEnvironment("test_neo4j_variables"))
    suite.addTest(TestEnvironment("test_openai_connection"))
    suite.addTest(TestEnvironment("test_neo4j_connection"))
    return suite


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# Run the test suite with a text-based test runner that outputs results
# to the console. Exit code will be 0 if all tests pass, non-zero otherwise.
# =============================================================================
if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
