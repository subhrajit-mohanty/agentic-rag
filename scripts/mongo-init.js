// MongoDB Initialization Script
// Creates initial database, collections, and indexes

// Switch to the enterprise_rag database
db = db.getSiblingDB('enterprise_rag');

// Create collections with validation
db.createCollection('knowledge_documents', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['document_id', 'title', 'content'],
      properties: {
        document_id: {
          bsonType: 'string',
          description: 'Unique document identifier'
        },
        title: {
          bsonType: 'string',
          description: 'Document title'
        },
        content: {
          bsonType: 'string',
          description: 'Document content'
        }
      }
    }
  }
});

db.createCollection('personas');
db.createCollection('query_logs');
db.createCollection('connectors');
db.createCollection('system_stats');

// Create indexes for knowledge_documents
db.knowledge_documents.createIndex({ 'document_id': 1 }, { unique: true });
db.knowledge_documents.createIndex({ 'title': 'text', 'content': 'text' });
db.knowledge_documents.createIndex({ 'metadata.category': 1 });
db.knowledge_documents.createIndex({ 'metadata.source': 1 });
db.knowledge_documents.createIndex({ 'created_at': -1 });
db.knowledge_documents.createIndex({ 'is_active': 1 });

// Create indexes for personas
db.personas.createIndex({ 'persona_id': 1 }, { unique: true });
db.personas.createIndex({ 'name': 1 });
db.personas.createIndex({ 'is_active': 1 });

// Create indexes for query_logs
db.query_logs.createIndex({ 'query_id': 1 }, { unique: true });
db.query_logs.createIndex({ 'created_at': -1 });
db.query_logs.createIndex({ 'persona_id': 1 });
db.query_logs.createIndex({ 'user_id': 1 });
db.query_logs.createIndex({ 'cache_hit': 1 });

// Create indexes for connectors
db.connectors.createIndex({ 'connector_id': 1 }, { unique: true });
db.connectors.createIndex({ 'connector_type': 1 });

// Create indexes for system_stats
db.system_stats.createIndex({ 'timestamp': -1 });

// Insert sample knowledge documents
db.knowledge_documents.insertMany([
  {
    document_id: 'doc_001',
    title: 'Employee Handbook - PTO Policy',
    content: 'Employees are entitled to 20 days of paid time off per year. Annual leave can be carried over up to 5 days to the next calendar year. Unused PTO beyond 5 days will be forfeited.',
    metadata: {
      source: 'SharePoint',
      filename: 'Employee_Handbook_v2.pdf',
      category: 'HR',
      tags: ['pto', 'leave', 'benefits']
    },
    chunk_index: 0,
    total_chunks: 1,
    created_at: new Date(),
    updated_at: new Date(),
    is_active: true
  },
  {
    document_id: 'doc_002',
    title: 'Project Ares Specifications',
    content: 'Project Ares is the internal code name for the 2025 cloud migration initiative. All legacy systems will be migrated to AWS. Timeline: Q1 2025 - Infrastructure setup, Q2 2025 - Application migration, Q3 2025 - Data migration, Q4 2025 - Decommission legacy.',
    metadata: {
      source: 'Confluence',
      filename: 'Project_Ares_Specs.md',
      category: 'Engineering',
      tags: ['project', 'aws', 'migration']
    },
    chunk_index: 0,
    total_chunks: 1,
    created_at: new Date(),
    updated_at: new Date(),
    is_active: true
  },
  {
    document_id: 'doc_003',
    title: 'Infrastructure Access Guide',
    content: 'S3 bucket access requires IAM role "Engineering-Role". All requests must be authenticated via SSO. Data encryption is mandatory for all S3 buckets. Use AWS KMS for key management. Contact DevOps for access requests.',
    metadata: {
      source: 'S3',
      filename: 'infra_access.md',
      category: 'DevOps',
      tags: ['s3', 'aws', 'security', 'access']
    },
    chunk_index: 0,
    total_chunks: 1,
    created_at: new Date(),
    updated_at: new Date(),
    is_active: true
  },
  {
    document_id: 'doc_004',
    title: 'Benefits Guide 2024',
    content: 'Health insurance benefits include medical, dental, and vision coverage. Employees can add dependents during open enrollment (November 1-30). The company covers 80% of premium costs. HSA contribution limit is $3,850 for individuals.',
    metadata: {
      source: 'SharePoint',
      filename: 'Benefits_Guide_2024.pdf',
      category: 'HR',
      tags: ['benefits', 'health', 'insurance']
    },
    chunk_index: 0,
    total_chunks: 1,
    created_at: new Date(),
    updated_at: new Date(),
    is_active: true
  },
  {
    document_id: 'doc_005',
    title: 'Engineering Code Standards',
    content: 'Code reviews are mandatory for all pull requests. At least two approvals required before merging to main branch. All code must pass CI/CD pipeline checks. Unit test coverage must be above 80%.',
    metadata: {
      source: 'GitHub',
      filename: 'CONTRIBUTING.md',
      category: 'Engineering',
      tags: ['code', 'review', 'standards']
    },
    chunk_index: 0,
    total_chunks: 1,
    created_at: new Date(),
    updated_at: new Date(),
    is_active: true
  }
]);

// Insert default personas
db.personas.insertMany([
  {
    persona_id: 'legal_analyst',
    name: 'Legal Analyst',
    system_prompt: 'You are a Legal Analyst AI assistant specializing in contract review, regulatory compliance (GDPR, SOC2, HIPAA), legal risk assessment, and policy interpretation. Always cite relevant policies or regulations. Flag potential compliance issues.',
    temperature: 0.1,
    allowed_tools: ['Retrieval'],
    allowed_categories: ['Legal', 'Compliance'],
    description: 'Specialized in legal document analysis',
    is_active: true,
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    persona_id: 'hr_specialist',
    name: 'HR Specialist',
    system_prompt: 'You are an HR Specialist AI assistant specializing in employee benefits, leave management, onboarding procedures, and performance reviews. Be empathetic and supportive. Always reference official HR policies.',
    temperature: 0.2,
    allowed_tools: ['Retrieval'],
    allowed_categories: ['HR', 'Benefits'],
    description: 'Expert in HR policies and procedures',
    is_active: true,
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    persona_id: 'tech_support',
    name: 'Tech Support',
    system_prompt: 'You are a Technical Support AI assistant specializing in AWS infrastructure, DevOps practices, internal tooling, and security best practices. Provide step-by-step instructions when applicable.',
    temperature: 0.1,
    allowed_tools: ['Retrieval'],
    allowed_categories: ['Engineering', 'DevOps'],
    description: 'Technical infrastructure specialist',
    is_active: true,
    created_at: new Date(),
    updated_at: new Date()
  }
]);

print('MongoDB initialization complete!');
print('Collections created: knowledge_documents, personas, query_logs, connectors, system_stats');
print('Sample documents inserted: 5');
print('Default personas inserted: 3');
