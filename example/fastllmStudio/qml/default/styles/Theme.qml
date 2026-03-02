pragma Singleton
import QtQuick

QtObject {
    // VS Dark palette
    readonly property color primaryColor: "#0E639C"
    readonly property color primaryHover: "#1177BB"
    readonly property color primaryPressed: "#0D5689"
    readonly property color accentColor: "#007ACC"

    // Surfaces — layered grays like VS Code
    readonly property color bgBase: "#1E1E1E"
    readonly property color bgSidebar: "#252526"
    readonly property color bgEditor: "#1E1E1E"
    readonly property color bgInput: "#3C3C3C"
    readonly property color bgCard: "#2D2D2D"
    readonly property color bgCardHover: "#333333"
    readonly property color bgToolbar: "#323233"
    readonly property color bgPopup: "#252526"

    // Sidebar
    readonly property color sidebarBg: "#333333"
    readonly property color sidebarItemHover: "#2A2D2E"
    readonly property color sidebarItemActive: "#37373D"
    readonly property color sidebarIndicator: "#007ACC"

    // Borders
    readonly property color border: "#3E3E42"
    readonly property color borderLight: "#474747"
    readonly property color borderFocus: "#007ACC"

    // Text
    readonly property color textPrimary: "#CCCCCC"
    readonly property color textSecondary: "#858585"
    readonly property color textBright: "#E0E0E0"
    readonly property color textDisabled: "#5A5A5A"
    readonly property color textLink: "#3794FF"
    readonly property color textOnAccent: "#FFFFFF"

    // Chat bubbles — VS-themed
    readonly property color userBubbleColor: "#264F78"
    readonly property color userBubbleText: "#D4D4D4"
    readonly property color assistantBubbleColor: "#333333"
    readonly property color assistantBubbleText: "#D4D4D4"
    readonly property color chatBackground: "#1E1E1E"

    // Status
    readonly property color successColor: "#4EC9B0"
    readonly property color errorColor: "#F14C4C"
    readonly property color warningColor: "#CCA700"
    readonly property color infoColor: "#3794FF"

    // Badge / tag
    readonly property color badgeBg: "#4D4D4D"
    readonly property color badgeText: "#CCCCCC"
    readonly property color runningBadgeBg: "#1B3A2D"
    readonly property color runningBadgeText: "#4EC9B0"
    readonly property color stoppedBadgeBg: "#3E3E42"
    readonly property color stoppedBadgeText: "#858585"

    // Scrollbar
    readonly property color scrollbar: "#424242"
    readonly property color scrollbarHover: "#4F4F4F"

    // Dimensions
    readonly property int sidebarWidth: 48
    readonly property int sidebarExpandedWidth: 220
    readonly property int toolbarHeight: 52
    readonly property int borderRadius: 4
    readonly property int borderRadiusLarge: 6
    readonly property int spacing: 10
    readonly property int spacingSmall: 6
    readonly property int spacingTiny: 4
    readonly property int paddingSmall: 6
    readonly property int paddingMedium: 12
    readonly property int paddingLarge: 20

    // Font
    readonly property string fontFamily: "Segoe UI, Ubuntu, Cantarell, sans-serif"
    readonly property int fontSizeTiny: 11
    readonly property int fontSizeSmall: 12
    readonly property int fontSizeMedium: 13
    readonly property int fontSizeLarge: 15
    readonly property int fontSizeTitle: 18
    readonly property int fontSizeHero: 24

    // Transitions
    readonly property int animDuration: 120
}
