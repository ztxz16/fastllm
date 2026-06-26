pragma Singleton
import QtQuick

QtObject {
    readonly property color primaryColor: "#6374F2"
    readonly property color primaryHover: "#7B8AFF"
    readonly property color primaryPressed: "#5364D8"
    readonly property color accentColor: "#6374F2"

    readonly property color bgBase: "#070A0D"
    readonly property color bgSidebar: "#11171D"
    readonly property color bgEditor: "#0D1318"
    readonly property color bgInput: "#101820"
    readonly property color bgCard: "#151C23"
    readonly property color bgCardHover: "#202A35"
    readonly property color bgToolbar: "#0B1116"
    readonly property color bgPopup: "#151C23"

    // Sidebar
    readonly property color sidebarBg: "#0B1014"
    readonly property color sidebarItemHover: "#202A35"
    readonly property color sidebarItemActive: "#202A35"
    readonly property color sidebarIndicator: "#6374F2"

    // Borders
    readonly property color border: "#24303B"
    readonly property color borderLight: "#344250"
    readonly property color borderFocus: "#6374F2"

    // Text
    readonly property color textPrimary: "#D8DEE6"
    readonly property color textSecondary: "#8E9AA8"
    readonly property color textBright: "#F4F7FB"
    readonly property color textDisabled: "#64717E"
    readonly property color textLink: "#46B9D8"
    readonly property color textOnAccent: "#FFFFFF"

    readonly property color userBubbleColor: "#1A222B"
    readonly property color userBubbleText: "#F4F7FB"
    readonly property color assistantBubbleColor: "#0D1318"
    readonly property color assistantBubbleText: "#D8DEE6"
    readonly property color chatBackground: "#0D1318"

    // Status
    readonly property color successColor: "#3FD08F"
    readonly property color errorColor: "#F05C68"
    readonly property color warningColor: "#D7A348"
    readonly property color infoColor: "#46B9D8"

    // Badge / tag
    readonly property color badgeBg: "#27313B"
    readonly property color badgeText: "#D8DEE6"
    readonly property color runningBadgeBg: "#163628"
    readonly property color runningBadgeText: "#3FD08F"
    readonly property color stoppedBadgeBg: "#27313B"
    readonly property color stoppedBadgeText: "#8E9AA8"

    // Scrollbar
    readonly property color scrollbar: "#344250"
    readonly property color scrollbarHover: "#455466"

    // Dimensions
    readonly property int sidebarWidth: 50
    readonly property int sidebarExpandedWidth: 252
    readonly property int toolbarHeight: 48
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
