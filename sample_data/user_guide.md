# CloudStore User Guide

## Getting Started

Welcome to CloudStore! This guide will help you get started with our cloud storage platform.

### Creating Your Account

1. Visit [cloudstore.example.com](https://cloudstore.example.com)
2. Click "Sign Up" button
3. Enter your email and create a password
4. Verify your email address
5. Complete your profile setup

The **UserManager** service handles all account-related operations. Your account information is securely stored and managed.

### Logging In

To access your files:

1. Go to the login page
2. Enter your credentials
3. Click "Log In"

The **AuthenticationService** validates your credentials and creates a secure session. You'll receive an authentication token that's valid for 24 hours.

## Managing Files

### Uploading Files

You can upload files in several ways:

- **Drag and Drop**: Simply drag files into the browser window
- **Upload Button**: Click the upload button and select files
- **Folder Upload**: Upload entire folders at once

When you upload a file:
1. The **FileManager** receives your file
2. The **QuotaManager** checks your available storage
3. The **StorageManager** saves the file securely
4. The **IndexService** indexes it for search

### Downloading Files

To download a file:
1. Navigate to the file
2. Click the download icon
3. Choose save location

The **FileManager** handles download requests and the **PermissionManager** verifies you have READ access.

### Organizing Files

Create folders to organize your files:
- Right-click and select "New Folder"
- Drag files into folders
- Use the move option in the file menu

## Sharing Files

### Share with Specific Users

1. Select the file or folder
2. Click "Share"
3. Enter the user's email
4. Choose permission level (View, Edit, or Admin)
5. Click "Send Invitation"

The **ShareService** manages all sharing operations. The **NotificationService** sends an email to the recipient.

### Creating Share Links

For sharing with anyone:

1. Select the file
2. Click "Get Link"
3. Set expiration and password (optional)
4. Copy and share the link

Share links are managed by the **ShareService** and **LinkManager**. You can revoke links anytime.

### Permission Levels

| Permission | Can View | Can Edit | Can Delete | Can Share |
|------------|----------|----------|------------|-----------|
| Viewer     | ✓        | ✗        | ✗          | ✗         |
| Editor     | ✓        | ✓        | ✗          | ✗         |
| Admin      | ✓        | ✓        | ✓          | ✓         |

The **PermissionManager** enforces these access controls across all operations.

## Search

Find files quickly using search:

- **Basic Search**: Type keywords in the search bar
- **Advanced Search**: Use filters for file type, date, size
- **Content Search**: Search within document contents

The **SearchService** provides fast, accurate results. Only files you have permission to access appear in results.

## Storage and Quotas

### Checking Your Usage

View your storage usage in Settings > Storage. You'll see:
- Total storage used
- Storage by file type
- Available space

The **QuotaManager** tracks your usage and enforces limits.

### Upgrading Storage

Need more space? Upgrade your plan:
- **Basic**: 15 GB free
- **Pro**: 100 GB for $9.99/month
- **Business**: 1 TB for $19.99/month

## Notifications

Stay informed with notifications:

- **Email**: Important updates and share invitations
- **Push**: Real-time mobile notifications
- **In-App**: Activity feed in the dashboard

Customize notifications in Settings > Notifications. The **NotificationService** handles all notification delivery.

## Security Features

### Two-Factor Authentication

Enable 2FA for extra security:
1. Go to Settings > Security
2. Click "Enable 2FA"
3. Scan QR code with authenticator app
4. Enter verification code

### Activity Log

Monitor account activity:
- View login history
- See file access logs
- Check sharing activity

The **AuditService** maintains comprehensive logs for security.

## Troubleshooting

### Common Issues

**Can't upload files?**
- Check your storage quota
- Verify file type is supported
- Check file size limits

**Can't access shared files?**
- Verify share link hasn't expired
- Check if you have correct permissions
- Contact the file owner

**Search not finding files?**
- Wait for indexing to complete
- Check your search terms
- Verify file permissions

### Getting Help

- **Help Center**: docs.cloudstore.example.com
- **Email Support**: support@cloudstore.example.com
- **Live Chat**: Available in-app 9 AM - 6 PM EST

## Keyboard Shortcuts

| Action          | Windows/Linux | Mac         |
|-----------------|---------------|-------------|
| Upload          | Ctrl+U        | Cmd+U       |
| New Folder      | Ctrl+Shift+N  | Cmd+Shift+N |
| Search          | Ctrl+F        | Cmd+F       |
| Delete          | Delete        | Delete      |
| Select All      | Ctrl+A        | Cmd+A       |
| Download        | Ctrl+D        | Cmd+D       |

---

Thank you for using CloudStore! For more information, visit our website or contact support.
