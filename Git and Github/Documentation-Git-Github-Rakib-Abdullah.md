# `Git and Github : Essential Commands for Industry Use`
## `Author : Rakib Abdullah`

## Introduction to Git and GitHub

### What is Git?

Git is a distributed version control system created by Linus Torvalds in 2005. It's designed to track changes in source code during software development and enables multiple developers to work together on non-linear development.

### What is Github

GitHub is a web-based hosting service for Git repositories. Launched in 2008 and acquired by Microsoft in 2018, GitHub provides the distributed version control of Git plus access control, bug tracking, feature requests, task management, continuous integration, and other collaborative features.

### Why Use Git and GitHub?

Together, Git and GitHub form the backbone of modern software development by enabling:

- **Collaboration**: Multiple developers can work on the same project simultaneously
- **History tracking**: Every change is recorded with author information
- **Rollback capability**: Revert to previous states if needed
- **Branching strategies**: Isolate work in progress from production code
- **Code reviews**: Improve code quality through peer review
- **Documentation**: Track why changes were made, not just what changed
- **Open-source contributions**: Facilitate participation in and maintenance of open-source projects
- **Portfolio building**: Showcase your work and contributions to potential employers

Nearly every software development team in industry uses Git for version control, with GitHub being the most popular hosting platform. Understanding these tools is an essential skill for developers, DevOps engineers, and many IT professionals.

## Initial Configuration

#### Set username and email (required for commits)
`git config --global user.name "My Name"`
`git config --global user.email "my.email@example.com"`

#### Set default editor
`git config --global core.editor "code --wait"  # For VS Code`

#### Set default branch name
`git config --global init.defaultBranch main`

## Starting a New Repository

#### Initialize a new Git repository
`git init`

#### Clone an existing repository
`git clone https://github.com/username/repository.git`

#### Clone a specific branch
`git clone -b branch-name https://github.com/username/repository.git`


## Basic Git Commands

#### Check status of working directory
`git status`

#### Add files to staging area
`git add filename`                # Add specific file
`git add directory/`              # Add all files in directory
`git add .`                         # Add all files in current directory
`git add -p`                       # Interactively add changes in hunks

#### Remove files
`git rm filename`                   # Remove file from Git and working directory
`git rm --cached filename`          # Stop tracking file but keep in working directory

