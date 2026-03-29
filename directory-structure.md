# Project Directory Structure

*Generated: 2025-09-03 09:24:51*

Summary: 15 directories, 58 files, 5 Python files

```
Mental_Health_AI/
.
├── backend
│   ├── chroma
│   │   ├── 55bd1ab2-645f-4718-99e8-b5278c1ac227
│   │   ├── d8e02def-f04d-4ff2-aa8e-177e2c9d20ea
│   │   └── e86c04f0-2745-4d90-a270-802ea757260c
│   ├── core
│   │   └── __pycache__
│   ├── main
│   │   ├── Mental_Health_Remedies
│   │   │   └── 379932df-a40c-4cf7-97f6-3930c108c162
│   │   ├── Mental_Health_Taboos_in_India
│   │   │   └── 26967f54-939a-4fe6-bce0-637056204cf1
│   │   └── __pycache__
│   └── vector_embeddings
└── docs
    └── images

17 directories
(base) manas@pop-os:~/PycharmProjects/Mental_Health_Helper_AI$ tree -a
.
├── backend
│   ├── chroma
│   │   ├── 55bd1ab2-645f-4718-99e8-b5278c1ac227
│   │   │   ├── data_level0.bin
│   │   │   ├── header.bin
│   │   │   ├── length.bin
│   │   │   └── link_lists.bin
│   │   ├── chroma.sqlite3
│   │   ├── d8e02def-f04d-4ff2-aa8e-177e2c9d20ea
│   │   │   ├── data_level0.bin
│   │   │   ├── header.bin
│   │   │   ├── length.bin
│   │   │   └── link_lists.bin
│   │   └── e86c04f0-2745-4d90-a270-802ea757260c
│   │       ├── data_level0.bin
│   │       ├── header.bin
│   │       ├── length.bin
│   │       └── link_lists.bin
│   ├── core
│   │   ├── mongo_schema.py
│   │   └── __pycache__
│   │       └── mongo_schema.cpython-312.pyc
│   ├── main
│   │   ├── 11Labs_testing.m4a
│   │   ├── apis.py
│   │   ├── auth.py
│   │   ├── chat_runner.py
│   │   ├── Mental_Health_Remedies
│   │   │   ├── 379932df-a40c-4cf7-97f6-3930c108c162
│   │   │   │   ├── data_level0.bin
│   │   │   │   ├── header.bin
│   │   │   │   ├── length.bin
│   │   │   │   └── link_lists.bin
│   │   │   └── chroma.sqlite3
│   │   ├── Mental_Health_Taboos_in_India
│   │   │   ├── 26967f54-939a-4fe6-bce0-637056204cf1
│   │   │   │   ├── data_level0.bin
│   │   │   │   ├── header.bin
│   │   │   │   ├── length.bin
│   │   │   │   └── link_lists.bin
│   │   │   └── chroma.sqlite3
│   │   ├── __pycache__
│   │   │   ├── chat_runner.cpython-312.pyc
│   │   │   └── voice_bridge.cpython-312.pyc
│   │   ├── redis_client.py
│   │   ├── reply.mp3
│   │   └── voice_bridge.py
│   ├── requirements.txt
│   └── vector_embeddings
│       ├── create_vector_embeddings.py
│       ├── guidelines-on-mental-health-promotive-and-preventive-interventions-for-adolescents-hat.pdf
│       ├── Mental_Health_Concerns_In_Indian_Population.pdf
│       ├── mental_health_india_youth_guide.pdf
│       ├── Mental_health_remedies_or_best_practices.pdf
│       ├── Mental_Health_Taboos_and_Issues_India_Compiled.pdf
│       └── Rhoads23TheBenefitsofYogaforDepression_AMeta-Analysis.pdf
├── directory-structure.md
├── docker-compose.yml
├── Dockerfile
├── docs
│   ├── chat_html.html
│   ├── images
│   │   ├── 24by7.jpg
│   │   ├── anonymous.jpg
│   │   ├── converse.jpg
│   │   ├── empathy.jpg
│   │   ├── encrypted.jpg
│   │   ├── goal.jpg
│   │   ├── instant.jpg
│   │   ├── monetize.jpg
│   │   ├── mood.jpg
│   │   ├── plans.jpg
│   │   └── private.jpg
│   ├── index.html
│   ├── login.html
│   ├── main.js
│   └── style.css
├── .env
├── .git
│   ├── branches
│   ├── COMMIT_EDITMSG
│   ├── config
│   ├── description
│   ├── HEAD
│   ├── hooks
│   │   ├── applypatch-msg.sample
│   │   ├── commit-msg.sample
│   │   ├── fsmonitor-watchman.sample
│   │   ├── post-update.sample
│   │   ├── pre-applypatch.sample
│   │   ├── pre-commit.sample
│   │   ├── pre-merge-commit.sample
│   │   ├── prepare-commit-msg.sample
│   │   ├── pre-push.sample
│   │   ├── pre-rebase.sample
│   │   ├── pre-receive.sample
│   │   ├── push-to-checkout.sample
│   │   ├── sendemail-validate.sample
│   │   └── update.sample
│   ├── index
│   ├── info
│   │   └── exclude
│   ├── logs
│   │   ├── HEAD
│   │   └── refs
│   │       ├── heads
│   │       │   └── main
│   │       └── remotes
│   │           └── origin
│   │               ├── HEAD
│   │               └── main
│   ├── objects
│   │   ├── 17
│   │   │   └── 189c562bc29a9244721fc4ddaea540bd001f63
│   │   ├── 39
│   │   │   └── 62b60b5477a462642dc3e86221e80bff4c979c
│   │   ├── 41
│   │   │   └── 87ccc109a46d4963640fd309862138de561c4d
│   │   ├── 6d
│   │   │   └── c1775a9d77e516c6d9822b594559c92920848a
│   │   ├── 6f
│   │   │   └── 33003a063e123cbf63ce61d3fe5debfa6cff16
│   │   ├── 7a
│   │   │   └── 09a46086d6bb4aa5346df2c668f3a65506ade8
│   │   ├── 88
│   │   │   └── 54be5be448433a884b63d7a90604c4fa55f57e
│   │   ├── 9c
│   │   │   └── f7577a62354943f69109f32babbbeab3cf3f68
│   │   ├── c4
│   │   │   └── 44f8947ade172e039edb54e8446115c49faf06
│   │   ├── e0
│   │   │   └── 7c6ae731935907d63ca4f80086021b67a14dae
│   │   ├── e7
│   │   │   └── cae7f8f213f8b5ebd3a9609c780eff4b7beae1
│   │   ├── f4
│   │   │   └── 8e6fc2e69dc367c73bee97ad12619b9f02678a
│   │   ├── info
│   │   └── pack
│   │       ├── pack-1f1756c1787fff35075668f6571c6fb24eb69188.idx
│   │       ├── pack-1f1756c1787fff35075668f6571c6fb24eb69188.pack
│   │       └── pack-1f1756c1787fff35075668f6571c6fb24eb69188.rev
│   ├── packed-refs
│   └── refs
│       ├── heads
│       │   └── main
│       ├── remotes
│       │   └── origin
│       │       ├── HEAD
│       │       └── main
│       └── tags
├── .gitattributes
├── .gitignore
├── .idea
│   ├── dictionaries
│   │   └── project.xml
│   ├── .gitignore
│   ├── inspectionProfiles
│   │   └── profiles_settings.xml
│   ├── Mental_Health_AI.iml
│   ├── misc.xml
│   ├── modules.xml
│   ├── vcs.xml
│   └── workspace.xml
├── README.md
├── render.yaml
└── requirements.txt

```