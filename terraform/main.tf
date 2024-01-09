provider "google" {
  project = "llm-exp-405305"
  region  = "us-central1"
  zone    = "us-central1-c"
}

resource "google_artifact_registry_repository" "inat" {
  provider               = google-beta
  project                = "llm-exp-405305"
  location               = "us"
  repository_id          = "wikibot"
  description            = "wikibot repo"
  format                 = "DOCKER"
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-tagged-release"
    action = "KEEP"
    condition {
      tag_state = "TAGGED"
    }
  }
  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 1 week
    }
  }
}
